from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddim.models.diffusion import get_timestep_embedding

from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from pg_modules.projector import F_RandomProj, RandomProj
from pg_modules.diffaug import DiffAugment


class SingleDisc(nn.Module):
        def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
                super().__init__()
                channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                                                256: 32, 512: 16, 1024: 8}

                # interpolate for start sz that are not powers of two
                if start_sz not in channel_dict.keys():
                        sizes = np.array(list(channel_dict.keys()))
                        start_sz = sizes[np.argmin(abs(sizes - start_sz))]
                self.start_sz = start_sz

                # if given ndf, allocate all layers with the same ndf
                if ndf is None:
                        nfc = channel_dict
                else:
                        nfc = {k: ndf for k, v in channel_dict.items()}

                # for feature map discriminators with nfc not in channel_dict
                # this is the case for the pretrained backbone (midas.pretrained)
                if nc is not None and head is None:
                        nfc[start_sz] = nc

                layers = []

                # Head if the initial input is the full modality
                if head:
                        layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                                           nn.LeakyReLU(0.2, inplace=True)]

                # Down Blocks
                DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
                while start_sz > end_sz:
                        layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
                        start_sz = start_sz // 2

                layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
                self.main = nn.Sequential(*layers)

        def forward(self, x, c=None):
                return self.main(x)


class SingleDiscCond(nn.Module):
        def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
                super().__init__()
                self.cmap_dim = cmap_dim

                # midas channels
                channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                                                256: 32, 512: 16, 1024: 8}

                # interpolate for start sz that are not powers of two
                if start_sz not in channel_dict.keys():
                        sizes = np.array(list(channel_dict.keys()))
                        start_sz = sizes[np.argmin(abs(sizes - start_sz))]
                self.start_sz = start_sz

                # if given ndf, allocate all layers with the same ndf
                if ndf is None:
                        nfc = channel_dict
                else:
                        nfc = {k: ndf for k, v in channel_dict.items()}

                # for feature map discriminators with nfc not in channel_dict
                # this is the case for the pretrained backbone (midas.pretrained)
                if nc is not None and head is None:
                        nfc[start_sz] = nc

                layers = []

                # Head if the initial input is the full modality
                if head:
                        layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                                           nn.LeakyReLU(0.2, inplace=True)]

                # Down Blocks
                DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
                while start_sz > end_sz:
                        layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
                        start_sz = start_sz // 2
                self.main = nn.Sequential(*layers)

                # additions for conditioning on class information
                self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
                self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
                self.embed_proj = nn.Sequential(
                        nn.Linear(self.embed.embedding_dim, self.cmap_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                )

        def forward(self, x, c):
                h = self.main(x)
                out = self.cls(h)

                # conditioning via projection
                cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
                out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

                return out

class SingleDiscTemb(nn.Module):
        def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, t_dim=512):
                super().__init__()
                self.t_dim = t_dim
                # midas channels
                channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                                                256: 32, 512: 16, 1024: 8}

                # interpolate for start sz that are not powers of two
                if start_sz not in channel_dict.keys():
                        sizes = np.array(list(channel_dict.keys()))
                        start_sz = sizes[np.argmin(abs(sizes - start_sz))]
                self.start_sz = start_sz

                # if given ndf, allocate all layers with the same ndf
                if ndf is None:
                        nfc = channel_dict
                else:
                        nfc = {k: ndf for k, v in channel_dict.items()}

                # for feature map discriminators with nfc not in channel_dict
                # this is the case for the pretrained backbone (midas.pretrained)
                if nc is not None and head is None:
                        nfc[start_sz] = nc

                layers = []
                emb_projs = []
                normalize_layers = []

                # Head if the initial input is the full modality
                if head:
                        layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                                           nn.LeakyReLU(0.2, inplace=True)]

                # Down Blocks
                DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
                while start_sz > end_sz:
                        layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
                        emb_proj_block = nn.Sequential(
                                nn.Linear(t_dim, nfc[start_sz//2]),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(nfc[start_sz//2], nfc[start_sz//2]*2),
                        )
                        nn.init.normal_(emb_proj_block[-1].weight, 0, 1e-4)
                        emb_projs.append(emb_proj_block)
                        normalize_layers.append(nn.GroupNorm(32, nfc[start_sz//2]))
                        start_sz = start_sz // 2

                self.main = nn.Sequential(*layers)
                self.emb_projs = nn.Sequential(*emb_projs)
                self.normalize = nn.Sequential(*normalize_layers)
                self.cls = conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False)


        def forward(self, x, prev_t, next_t):
                # h = self.main(x)
                # out = self.cls(h)

                prev_temb = get_timestep_embedding(prev_t, self.t_dim//2)
                next_temb = get_timestep_embedding(next_t, self.t_dim//2)
                h = x
                for layer, emb_proj, norms in zip(self.main, self.emb_projs, self.normalize):
                        h = layer(h)
                        h_bias_scale = emb_proj(torch.cat([prev_temb, next_temb], dim=-1))[...,None,None]
                        h_scale, h_bias = torch.chunk(h_bias_scale, 2, 1)
                        h = norms(h) * (h_scale+1) + h_bias

                # conditioning via projection
                out = self.cls(h)

                return out

class MultiScaleD(nn.Module):
        def __init__(
                self,
                channels,
                resolutions,
                num_discs=1,
                proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
                cond=0,
                tcond=0,
                separable=False,
                patch=False,
                **kwargs,
        ):
                super().__init__()

                assert num_discs in [1, 2, 3, 4]

                # the first disc is on the lowest level of the backbone
                self.disc_in_channels = channels[:num_discs]
                self.disc_in_res = resolutions[:num_discs]
                Disc = SingleDiscCond if cond else SingleDisc
                if tcond:
                        Disc = SingleDiscTemb

                mini_discs = []
                for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
                        start_sz = res if not patch else 16
                        mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],
                self.mini_discs = nn.ModuleDict(mini_discs)

        # def forward(self, features, t1, t2):
        #         all_logits = []
        #         for k, disc in self.mini_discs.items():
        #                 all_logits.append(disc(features[k], t1, t2).view(features[k].size(0), -1))

        #         all_logits = torch.cat(all_logits, dim=1)
        #         return all_logits

        def forward(self, features, t1, t2):
                all_logits = []
                for k, disc in self.mini_discs.items():
                        all_logits.append(disc(features[k], t1, t2).view(features[k].size(0), -1))

                all_logits = torch.cat(all_logits, dim=1)
                return all_logits


class ProjectedDiscriminator(torch.nn.Module):
        def __init__(
                self,
                diffaug=True,
                interp224=True,
                backbone_kwargs={},
                **kwargs
        ):
                super().__init__()
                self.diffaug = diffaug
                self.interp224 = interp224
                self.feature_network = F_RandomProj(**backbone_kwargs)
                self.discriminator = MultiScaleD(
                        channels=self.feature_network.CHANNELS,
                        resolutions=self.feature_network.RESOLUTIONS,
                        **backbone_kwargs,
                )

        def train(self, mode=True):
                self.feature_network = self.feature_network.train(False)
                self.discriminator = self.discriminator.train(mode)
                return self

        def eval(self):
                return self.train(False)

        def forward(self, x, c):
                if self.diffaug:
                        x = DiffAugment(x, policy='color,translation,cutout')

                if self.interp224:
                        x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

                features = self.feature_network(x)
                logits = self.discriminator(features, c)

                return logits

class ProjectedPairedDiscriminator(torch.nn.Module):
        def __init__(
                self,
                diffaug=False,
                interp224=True,
                backbone_kwargs={},
                feature_network=None,
                **kwargs
        ):
                super().__init__()
                self.diffaug = diffaug
                self.interp224 = interp224
                #self.feature_network = feature_network
                #self.proj_network = RandomProj(im_res=32, cout=64, proj_type=2, 
                #                channels=[128, 256, 256, 256], resolutions=[32, 16, 8, 4])

                self.feature_network = F_RandomProj(**backbone_kwargs)

                self.discriminator = MultiScaleD(
                        channels=[f*2 for f in self.feature_network.CHANNELS],
                        resolutions=self.feature_network.RESOLUTIONS,
                        tcond=1,
                        **backbone_kwargs,
                )

        def train(self, mode=True):
                self.feature_network = self.feature_network.train(False)
                #self.proj_network = self.proj_network.train(False)
                self.discriminator = self.discriminator.train(mode)
                return self

        def eval(self):
                return self.train(False)
        
        def forward(self, x, t):
                #if self.diffaug:
                #       x = DiffAugment(x, policy='color,translation,cutout')

                #if self.interp224:
                x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
                feature_list = self.feature_network(x)

                return feature_list

        def disc(self, z1, z2, t1, t2):
                paired_z = {k: torch.cat([z1[k], z2[k]], axis=1) for k in z1.keys()}
                logits = self.discriminator(paired_z, t1, t2)

                return logits

#feature_list.reverse()
#proj_feature = self.proj_network(feature_list)
#self.discriminator = MultiScaleD(
#                       channels=[ch*2 for ch in self.proj_network.channels],
#                       resolutions = self.proj_network.resolutions,
#                       tcond=1,
#                       **backbone_kwargs,
#               )

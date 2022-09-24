from functools import partial
import numpy as np
from ddim.models.diffusion import get_timestep_embedding
from pg_modules.diffaug_pair import DiffAugment_pair
import torch
import torch.nn as nn
import torch.nn.functional as F

from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from pg_modules.projector import F_RandomProj
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

    def forward(self, x, c, scale):
        return self.main(x)

class SingleDiscScond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, semb_ch=128):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz
        self.temb_ch = semb_ch

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
        semb_layer = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]
            # semb_layer += [nn.Linear(self.temb_ch, nfc[256]*2)]
            semb_layer += [nn.Linear(self.temb_ch, nfc[256])]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            # semb_layer += [nn.Linear(self.temb_ch, nfc[start_sz//2]*2)]
            semb_layer += [nn.Linear(self.temb_ch, nfc[start_sz//2])]
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.ModuleList(layers)
        self.semb = nn.ModuleList(semb_layer)

    def forward(self, x, c, semb):
        h = x
        for main_layer, semb_layer in zip(self.main, self.semb):
            h = main_layer(h)
            # s_scale, s_bias = torch.chunk(semb_layer(semb), 2, 1)
            s_bias = semb_layer(semb)
            h = h + s_bias[...,None,None]
        return self.main[-1](h)


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


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        scond=0,
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
        Disc = SingleDiscScond if scond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, scale=None):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c, scale).view(features[k].size(0), -1))

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


class ProjectedPairDiscriminator(torch.nn.Module):
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
        self.temb_ch = 512
        self.scale_proj = nn.Sequential(*[
            nn.Linear(self.temb_ch, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128)
        ])

        self.discriminator = MultiScaleD(
            #channels=[f*2 for f in self.feature_network.CHANNELS],
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            scond=1,
            **backbone_kwargs,
        )
        self.pair_discriminator = MultiScaleD(
            channels=[f + 32 for f in self.feature_network.CHANNELS],
            resolutions=self.feature_network.RESOLUTIONS,
            scond=1,
            **backbone_kwargs,
        )
        self.pair_norm = nn.ModuleDict({
            str(i): nn.InstanceNorm2d(f, affine=False)
            for i, (f, r) in enumerate(zip([24, 40, 112, 320], [112, 56, 28, 14]))
        })


    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        self.scale_proj = self.scale_proj.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, high, low, c, scale=None):
        #high, low = DiffAugment_pair(high, low, policy='color,translation,cutout')
        high = DiffAugment(high, policy='color,translation,cutout')

        # interp = max(224, high.shape[2])
        high = F.interpolate(high, 224, mode='bilinear', align_corners=False)
        #low = F.interpolate(low, interp, mode='bilinear', align_corners=False)

        #x = torch.cat([high, low])
        x = high
        features = self.feature_network(x)
        #features = {k:torch.cat(torch.chunk(v, 2, 0), 1) for k,v in features.items()}
        if scale == None:
            scale = torch.ones((len(high),), device=high.device)
        semb = get_timestep_embedding(scale.squeeze()*1000, self.temb_ch)
        semb = self.scale_proj(semb)

        logits = self.discriminator(features, c, semb)

        return logits

    def pair_disc(self, x1, x2, scale=None, c=None, real=False):
        # x = torch.cat([x1, x2])
        x = x1
        x = DiffAugment(x, policy='color,translation,cutout')

        #if self.interp224:
        x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
        
        features = self.feature_network.pretrain_forward(x)
        shuffle_idx = torch.randperm(len(x), device=x.device)
        inst_mean = {k: v.mean([2,3], keepdims=True) for k,v in features.items()}
        inst_var = {k: (v - v.mean([2,3], keepdims=True)).square().mean([2,3], keepdims=True) + 1e-6 for k,v in features.items()}
        mix_rate = torch.rand((len(x), 1, 1, 1), device=x.device)
        new_mean = {k: inst_mean[k] * mix_rate + inst_mean[k][shuffle_idx] * (1 - mix_rate) for k in inst_mean.keys()}
        new_var = {k: inst_var[k] * mix_rate + inst_var[k][shuffle_idx] * (1 - mix_rate) for k in inst_mean.keys()}

        features = {k: (self.pair_norm[k](v) + new_mean[k]) * new_var[k].sqrt()  for k,v in features.items()}
        features = self.feature_network.proj_forward(features)

        pair_features = {}
        semb = get_timestep_embedding(scale.squeeze() * 1000, self.temb_ch)
        semb = self.scale_proj(semb)

        # scale = scale.reshape(-1, 1, 1, 1)
        for k, v in features.items():
            x1_feat = v
            x2_feat = torch.nn.functional.interpolate(x2, size=(x1_feat.shape[2], x1_feat.shape[2]), mode='bilinear')
            # x1_feat, x2_feat = torch.chunk(v, 2, 0)
            # x1_feat = x1_feat * scale.sqrt() + torch.randn_like(x1_feat) * (1 - scale).sqrt()
            # ch_proj = torch.randn((x1_feat.shape[1], x1_feat.shape[1]), device=x1_feat.device)
            pair_features[k] = torch.cat([x1_feat, x2_feat], 1)

        logits = self.pair_discriminator(pair_features, c, semb)

        return logits

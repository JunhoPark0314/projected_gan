# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import torch.nn as nn
from ddim.models.diffusion import get_timestep_embedding
from pg_modules.blocks import (AttnBlock, BlockBig, DownBlock, InitLayer, UpBlockBig, UpBlockBigCond, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d)
import torch

from pg_modules.diffaug import DiffAugment

def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API

class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.temb_ch =512

        # channel multiplier
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)
        self.h_init = InitLayer(z_dim, channel=nfc[8], sz=4)
        # self.h_proj = nn.Conv2d(32, nfc[16], 1)
        # self.h_proj = nn.parameter.Parameter(torch.randn((32, nfc[8])))
        # self.h_gain = 1 / np.sqrt(32)
        # self.se_proj =SEBlock(nfc[4], nfc[8])
        self.scale_proj = nn.Sequential(*[
            nn.Linear(self.temb_ch, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.scale_8 = nn.Linear(512, nfc[8] * 2)
        # self.scale_16 = nn.Linear(512, nfc[16])

        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.h_proj = BlockBig(32, nfc[8])
        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_proj = SEBlock(nfc[8], nfc[8])
        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)
        self.to_small = conv2d(nfc[32], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, h, input, c, scale, **kwargs):
            # map noise to hypersphere as in "Progressive Growing of GANS"
            input = normalize_second_moment(input[:, 0])
            temb = get_timestep_embedding(scale.squeeze() * 100, self.temb_ch)
            temb = self.scale_proj(temb)
            t_scale_8, t_bias_8 = torch.chunk(self.scale_8(temb), 2, 1)
            t_scale_8 = t_scale_8.sigmoid()[...,None,None]
            # t_bias_16 = self.scale_16(temb)

            feat_4 = self.init(input) 
            h_proj = self.h_proj(h)
            # h_proj = self.se_proj(feat_4 * t_scale[...,None,None] + t_bias[...,None,None], h_proj)
            # feat_8 = self.feat_8(feat_4) + torch.einsum('bchw,ck->bkhw', h, self.h_proj * self.h_gain) + t_bias[...,None,None]
            feat_8 = self.feat_8(feat_4) #+ t_bias[...,None,None]

            feat_8 = self.se_proj(h_proj + t_bias_8[...,None,None], feat_8) * (1 - t_scale_8).sqrt() + h_proj * t_scale_8.sqrt()

            feat_16 = self.feat_16(feat_8)
            feat_32 = self.feat_32(feat_16)

            feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
            feat_128 = self.se_128(feat_8,  self.feat_128(feat_64))

            if self.img_resolution >= 128:
                feat_last = feat_128

            if self.img_resolution >= 256:
                feat_last = self.se_256(feat_16, self.feat_256(feat_last))

            if self.img_resolution >= 512:
                feat_last = self.se_512(feat_32, self.feat_512(feat_last))

            if self.img_resolution >= 1024:
                feat_last = self.feat_1024(feat_last)

            out = self.to_big(feat_last)
            return out, self.to_small(feat_32), h


class FastganSynthesisCond(nn.Module):
    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256, num_classes=1000, lite=False):
        super().__init__()

        self.z_dim = z_dim
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125, 2048:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.img_resolution = img_resolution

        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond

        self.feat_8   = UpBlock(nfc[4], nfc[8], z_dim)
        self.feat_16  = UpBlock(nfc[8], nfc[16], z_dim)
        self.feat_32  = UpBlock(nfc[16], nfc[32], z_dim)
        self.feat_64  = UpBlock(nfc[32], nfc[64], z_dim)
        self.feat_128 = UpBlock(nfc[64], nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.embed = nn.Embedding(num_classes, z_dim)

    def forward(self, input, c, update_emas=False):
        c = self.embed(c.argmax(1))

        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4, c)
        feat_16 = self.feat_16(feat_8, c)
        feat_32 = self.feat_32(feat_16, c)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32, c))
        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64, c))

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, c))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, c))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c)

        return self.to_big(feat_last)


import numpy as np
class Encoder(nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        hidden_ch = 128,
        z_dim=256,
		out_ch=32,
    ):
        super().__init__()
        self.out_ch = out_ch
        num_layer = int(np.log2(img_resolution)) - 3
        self.layers = []
        in_ch = img_channels
        hidden_ch = hidden_ch
        for i in range(num_layer):
            self.layers.append(DownBlock(in_ch, hidden_ch))
            if i == (num_layer - 2):
                self.layers.append(AttnBlock(hidden_ch))
            in_ch = hidden_ch
        self.layers.append(nn.Conv2d(hidden_ch, out_ch, 1, 1))
        self.layers.append(nn.InstanceNorm2d(out_ch, affine=False))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, z, c, **kwargs):
        # x = DiffAugment(x, policy='color,cutout')

        enc = self.layers(x)
        scale = torch.rand((x.shape[0], 1, 1, 1), device=x.device)
        scale = kwargs.get("scale", scale)
        temp_min, temp_max = kwargs.get("temp_min", 0.03), kwargs.get("temp_max", 1.0)
        scale = scale.clip(min=temp_min, max=temp_max)
        noise = torch.randn_like(enc, device=x.device)
        # ch_proj = torch.randn((self.out_ch, self.out_ch), device=x.device)
        # enc = torch.einsum("bchw,ck->bkhw",enc, ch_proj)

        out = (enc * scale.sqrt() + noise * (1 - scale).sqrt())
        return out, z.unsqueeze(1), scale.squeeze()

class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Mapping and Synthesis Networks
        self.mapping = Encoder(img_resolution, img_channels, z_dim)
        Synthesis = FastganSynthesisCond if cond else FastganSynthesis
        self.synthesis = Synthesis(ngf=ngf, z_dim=z_dim, nc=img_channels, img_resolution=img_resolution, **synthesis_kwargs)

    def forward(self, x, z, c, return_small=False, **kwargs):
        h, w, scale = self.mapping(x, z, c, **kwargs)
        high_res, low_res, h_proj = self.synthesis(h, w, c, scale)
        if return_small:
            return high_res, low_res, scale, h_proj.detach()
        return high_res

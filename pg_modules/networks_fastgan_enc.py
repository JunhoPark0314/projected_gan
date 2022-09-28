# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import torch.nn as nn
from torch_utils.misc import compute_alpha, get_timestep_embedding, get_beta_schedule
from pg_modules.blocks import (AttnBlock, BlockBig, DownBlock, InitLayer, UpBlockBig, UpBlockBigCond, UpBlockDenoise, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d)
import torch

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
        self.out_ch = 64

        # channel multiplier
        # nfc_multi = {2: 8, 4:8, 8:4, 16:4, 32:2, 64:2, 128:1, 256:0.5,
        #              512:0.25, 1024:0.125, 2048:0.125}
        # nfc_multi = {2: 16, 4:8, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
        #              512:0.25, 1024:0.125, 2048:0.125}
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125, 2048:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[4], sz=4)
        # self.h_init = InitLayer(z_dim, channel=nfc[2], sz=4)
        # self.h_proj = nn.Conv2d(32, nfc[16], 1)
        # self.h_proj = nn.parameter.Parameter(torch.randn((32, nfc[8])))
        # self.h_gain = 1 / np.sqrt(32)
        # self.se_proj =SEBlock(nfc[4], nfc[8])
        self.scale_proj = nn.Sequential(*[
            nn.Linear(self.temb_ch, 128),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.scale_bias_16 = nn.Sequential(*[
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        ])
        self.scale_bias_4 = nn.Sequential(*[
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, nfc[8]),
        ])

        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.h_down_8 = DownBlock(self.out_ch, nfc[16], norm='group')
        self.h_down_4 = DownBlock(nfc[16], nfc[8], norm='group')

        self.h_up_4 = UpBlockDenoise(nfc[8], nfc[16])
        self.h_up_8 = UpBlockDenoise(nfc[16], self.out_ch)
        self.h_zero = nn.Conv2d(self.out_ch, self.out_ch, 1, 1)
        nn.init.constant_(self.h_zero.weight, 0)

        # self.feat_proj = BlockBig(nfc[16], nfc[16])
        self.h_proj = BlockBig(self.out_ch, nfc[16])
        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])

        self.proj_16 = nn.Conv2d(nfc[16]*2, nfc[16], 3, 1, 1)

        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_init = SEBlock(self.out_ch, nfc[8])
        self.se_h4 = SEBlock(self.out_ch, nfc[4])
        self.se_h8 = SEBlock(self.out_ch, nfc[8])

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)
        # self.to_small = conv2d(nfc[32], nc, 3, 1, 1, bias=True)
        # self.to_main = conv2d(32, nfc[16], 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, h, input, c, scale, alpha, **kwargs):
            # map noise to hypersphere as in "Progressive Growing of GANS"
            input = normalize_second_moment(input[:, 0])
            # h = normalize_second_moment(h)

            temb = get_timestep_embedding(scale.squeeze() * 1000, self.temb_ch)
            temb = self.scale_proj(temb)
            t_scale_16 = self.scale_bias_16(temb)[...,None,None].sigmoid()
            t_bias_4 = self.scale_bias_4(temb)[...,None,None]

            # denoise step
            h_8 = self.h_down_8(h)
            h_4 = self.h_down_4(h_8)
            h_4 = self.h_up_4(h_4 + t_bias_4)
            h_8 = self.h_zero(self.h_up_8(h_8 + h_4)) * (1 - alpha).sqrt()
            denoised_h = (h + h_8) / alpha.sqrt()

            feat_4 = self.se_h4(denoised_h, self.init(input))
            feat_8 = self.se_h8(denoised_h, self.feat_8(feat_4))
            # feat_16 = self.feat_16(feat_8) * t_scale_16.sqrt() + self.h_proj(denoised_h) * (1 - t_scale_16).sqrt()
            h_proj = self.h_proj(denoised_h) * (1 - t_scale_16).sqrt()
            feat_16 = self.feat_16(feat_8) * t_scale_16.sqrt()
            feat_16 = self.proj_16(torch.cat([feat_16, h_proj], dim=1))

            # feat_16 = self.feat_proj(feat_16)
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
            return out, None, (h.detach() + h_8) / alpha.sqrt()


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
        hidden_ch = 64,
        z_dim=256,
		out_ch=64,
    ):
        super().__init__()
        self.layers = []
        # self.layers.append(nn.InstanceNorm2d(out_ch, affine=False))
        # self.layers.append(nn.BatchNorm2d(out_ch))
        self.layers.append(nn.GroupNorm(16, out_ch, affine=False))
        self.layers = nn.Sequential(*self.layers)
        self.out_ch = out_ch

        # self.flt_out = nn.Sequential(*[
        #     nn.Linear(8*8*out_ch, z_dim*2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(z_dim*2, z_dim)
        # ])
        # self.out_norm = nn.LayerNorm((z_dim,), elementwise_affine=False)
        # self.out_norm = nn.InstanceNorm2d(out_ch, affine=False)
        # self.out_norm = nn.LayerNorm((out_ch,), affine=False)

        self.num_timesteps = 1000
        betas = get_beta_schedule(
            beta_schedule="linear", 
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000)
        self.register_buffer("betas", torch.tensor(betas).float())

    def forward(self, x, z, c, **kwargs):
        enc = self.encode(x)
        enc = self.layers(enc)

        n = len(enc)
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to(enc.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        temp_min, temp_max = kwargs.get("temp_min", 0.0), kwargs.get("temp_max", 1.0)
        temp_max = min(temp_max, 0.4)
        temp_min = min(temp_min, temp_max)
        t = (t * (temp_max - temp_min) + self.num_timesteps * temp_min).floor().long()
        alpha = compute_alpha(self.betas, t)

        noise = torch.randn_like(enc, device=x.device)
        out = (enc * alpha.sqrt() + noise * (1 - alpha).sqrt())
        # out = enc
        return out, z.unsqueeze(1), t/1000, enc, alpha
    
    def sample_noised(self, enc, **kwargs):
        n = len(enc)
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to(enc.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        temp_min, temp_max = kwargs.get("temp_min", 0.0), kwargs.get("temp_max", 1.0)
        temp_max = min(temp_max, 0.4)
        temp_min = min(temp_min, temp_max)
        t = (t * (temp_max - temp_min) + self.num_timesteps * temp_min).floor().long()
        alpha = compute_alpha(self.betas, t)

        noise = torch.randn_like(enc, device=enc.device)
        out = (enc * alpha.sqrt() + noise * (1 - alpha).sqrt())
        # out = enc
        return out, t/1000, alpha

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
        h, w, scale, enc, alpha = self.mapping(x, z, c, **kwargs)
        high_res, low_res, denoised_h = self.synthesis(h, w, c, scale, alpha)
        if return_small:
            return high_res, low_res, scale, denoised_h, enc
        return high_res

# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import torch.nn as nn
from ddim.models.diffusion import FullyConnectedLayer
from torch_utils.misc import compute_alpha, get_timestep_embedding, get_beta_schedule
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
        self.h_init = InitLayer(z_dim, channel=32, sz=16)
        # self.h_proj = nn.Conv2d(32, nfc[16], 1)
        # self.h_proj = nn.parameter.Parameter(torch.randn((32, nfc[8])))
        # self.h_gain = 1 / np.sqrt(32)
        # self.se_proj =SEBlock(nfc[4], nfc[8])
        self.scale_proj = nn.Sequential(*[
            nn.Linear(self.temb_ch, 128),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.scale_bias = nn.Sequential(*[
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, nfc[8]),
        ])
        self.scale_16 = nn.Sequential(*[
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        ])

        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.h_proj = BlockBig(320, nfc[8])
        self.feat_proj = BlockBig(nfc[16], nfc[16])
        # self.h_ch_proj = BlockBig(32, 32)
        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_init = SEBlock(nfc[8], nfc[8])
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

    def forward(self, h, input, c, scale, **kwargs):
            # map noise to hypersphere as in "Progressive Growing of GANS"
            input = normalize_second_moment(input[:, 0])
            temb = get_timestep_embedding(scale.squeeze() * 1000, self.temb_ch)
            temb = self.scale_proj(temb)
            # t_16 = self.scale_16(temb)
            # t_scale_16 = t_16[:,:1].sigmoid()[...,None,None]
            # t_bias_16 = t_16[:,1:][...,None,None]
            t_bias = self.scale_bias(temb)

            feat_4 = self.init(input) 
            # feat_h_init = self.h_init(input + t_bias)
            feat_8 = self.feat_8(feat_4)
            # h_proj = self.h_proj(h)
            main_feat_8 = self.se_init(feat_8, self.h_proj(h)) + t_bias[...,None,None]
            # main_feat_8 = feat_8

            feat_16 = self.feat_16(main_feat_8) #* (1 - t_scale_16).sqrt() + h_proj * t_scale_16.sqrt()

            feat_16 = self.feat_proj(feat_16)
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
            return out, None, h


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
		out_ch=32,
    ):
        super().__init__()
        in_ch = 320
        num_layer = 2
        down_layers = []
        for i in range(num_layer):
            down_layers.append(DownBlock(in_ch, hidden_ch))
            in_ch = hidden_ch
        self.down_layers = nn.ModuleList(down_layers)

        self.middle_layer =AttnBlock(hidden_ch)

        up_layers = []
        for i in range(num_layer):
            up_layers.append(UpBlockSmall(hidden_ch, hidden_ch))
        self.up_layers = nn.ModuleList(up_layers)
        self.out = nn.Conv2d(hidden_ch, 320, 3, 3)
        nn.init.constant(self.out.weight, 0)

        self.out_norm = nn.InstanceNorm2d(in_ch, affine=False)
        self.num_timesteps = 1000

        betas = get_beta_schedule(
            beta_schedule="linear", 
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000)
        self.register_buffer("betas", torch.tensor(betas).float())

    def forward(self, x, z, c, **kwargs):
        # x = DiffAugment(x, policy='color,cutout')

        # enc = self.layers(x)
        # enc = self.out(enc.reshape(-1, 32, 16*16).permute(0, 2, 1).reshape(-1, 32)).reshape(-1, 16*16, 32).permute(0, 2, 1).reshape(-1, 32, 16, 16)
        # enc = self.out_norm(enc)
        x = torch.nn.functional.interpolate(x, 224, mode='bilinear', align_corners=False)
        enc = self.feature_network.pretrain_forward(x)['3']
        enc = self.out_norm(enc)
        # enc = self.out(enc)
        enc = torch.nn.functional.interpolate(enc, 8, mode='bilinear', align_corners=False)

        n = len(enc)
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to(enc.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        temp_min, temp_max = kwargs.get("temp_min", 0.0), kwargs.get("temp_max", 1.0)
        # temp_max = min(temp_max, 0.5)
        # temp_min = min(temp_min, temp_max)
        t = (t * (temp_max - temp_min) + self.num_timesteps * temp_min).floor().long()
        alpha = compute_alpha(self.betas, t)

        noise = torch.randn_like(enc, device=x.device)
        out = (enc * alpha.sqrt() + noise * (1 - alpha).sqrt())

        h = out
        hs = []
        for down in self.down_layers:
            h = down(h)
            hs.append(h)
        h = self.middle_layer(hs[-1])
        for up in self.up_layers:
            h = up(h + hs.pop())
        assert not hs

        denoised_out = (out + self.out(h) * (1 - alpha).sqrt()) / alpha.sqrt()
        return denoised_out, enc, z.unsqueeze(1), t/1000

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
        h, enc, w, scale = self.mapping(x, z, c, **kwargs)
        high_res, low_res, _ = self.synthesis(h, w, c, scale)
        if return_small:
            return high_res, low_res, scale, h, enc
        return high_res

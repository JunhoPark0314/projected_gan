# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import torch.nn as nn
from pg_modules.blocks import (InitLayer, UpBlockBig, UpBlockBigCond, UpBlockBigUnet, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d, InitLayerUnet)
from pg_modules.diffusion import Encoder


def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c=None, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API


class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False, ddim_config=None):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.unet_dim = [256, 256, 256, 128]

        # channel multiplier
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayerUnet(z_dim, channel=nfc[2], sz=4, u_ch=self.unet_dim[0])

        # UpBlock = UpBlockSmall if lite else UpBlockBig
        UpBlock = UpBlockBigUnet

        self.feat_8   = UpBlock(nfc[4], nfc[8], self.unet_dim[1])
        self.feat_16  = UpBlock(nfc[8], nfc[16], self.unet_dim[2], last=2)
        self.feat_32  = UpBlock(nfc[16], nfc[32], self.unet_dim[3])

        self.se_16    = SEBlock(nfc[4], nfc[16])
        self.se_32    = SEBlock(nfc[8], nfc[32])

        # self.feat_64  = UpBlock(nfc[32], nfc[64])
        # self.feat_128 = UpBlock(nfc[64], nfc[128])
        # self.feat_256 = UpBlock(nfc[128], nfc[256])

        # self.se_64  = SEBlock(nfc[4], nfc[64])
        # self.se_128 = SEBlock(nfc[8], nfc[128])
        # self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        # if img_resolution > 256:
        #     self.feat_512 = UpBlock(nfc[256], nfc[512])
        #     self.se_512 = SEBlock(nfc[32], nfc[512])
        # if img_resolution > 512:
        #     self.feat_1024 = UpBlock(nfc[512], nfc[1024])
    
        self.encoder = Encoder(ddim_config)

    def forward(self, x, t, w, **kwargs):
        # map noise to hypersphere as in "Progressive Growing of GANS"
        hs = self.encoder(x, t)
        hs.reverse()
        st = 0
        input = normalize_second_moment(w[:, 0])

        feat_4 = self.init(input, hs[st:st+3])
        st += 3
        feat_8 = self.feat_8(feat_4, hs[st:st+3])
        st += 3
        feat_16 = self.se_16(feat_4, self.feat_16(feat_8, hs[st:st+3]))
        st += 3
        feat_last = self.se_32(feat_8, self.feat_32(feat_16, hs[st:st+3]))

        return self.to_big(feat_last)

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
        synthesis_kwargs={},
        ddim_config={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Mapping and Synthesis Networks
        self.mapping = DummyMapping()  # to fit the StyleGAN API
        Synthesis = FastganSynthesis
        self.synthesis = Synthesis(ngf=ngf, z_dim=z_dim, nc=img_channels, img_resolution=img_resolution, ddim_config=ddim_config, **synthesis_kwargs)

    def forward(self, x, t, z, **kwargs):
        w = self.mapping(z)
        img = self.synthesis(x, t, w)
        return img

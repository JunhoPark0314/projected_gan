# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

class DDIM_Loss(Loss):
    def __init__(self, device, G_ema, DDIM, diffusion,**kwargs):
        super().__init__()
        self.device = device
        self.G_ema = G_ema
        self.DDIM = DDIM
        self.diffusion = diffusion

    def run_G(self, real_img, update_emas=False):
        t = self.diffusion.sample_timestep(len(real_img))
        h = self.G_ema.mapping.layers(real_img)
        h_noised, eps, alpha = self.diffusion.sample_noised(h, t)
        trg_eps = self.DDIM(h_noised, t.float())
        return trg_eps, eps, alpha

    def accumulate_gradients(self, phase, real_img, cur_nimg):
        assert phase in ['DDIMboth']
        # Gmain: Maximize logits for generated images.
        with torch.autograd.profiler.record_function('Gmain_forward'):
            pred_eps, trg_eps, alpha = self.run_G(real_img)
            loss_Rec = (pred_eps - trg_eps).square().sum([1,2,3]).mean()
            # Logging
            training_stats.report('Loss/DDIM/loss_rec', loss_Rec)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Rec.backward()

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()


class ProjectedGANPairLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.warmup_nimg = 20 * 2**12

    def run_G(self, real_img, z, c, update_emas=False, temp_max=1.0):
        # if use_ema:
        #     h, ws, scale, enc = self.G_ema.mapping(real_img, z, c, temp_max=temp_max, update_emas=update_emas)
        # else:
        h, ws, scale, enc, alpha = self.G.mapping(real_img, z, c, temp_max=temp_max, update_emas=update_emas)
        high, low, _ = self.G.synthesis(h, ws, c, scale, alpha, update_emas=False)
        return high, low, scale.squeeze()[:,None], h, enc, alpha

    def run_D(self, x, h, c, scale, alpha, blur_sigma=0, update_emas=False):
        logits = self.D(x, h.detach(), c, scale, alpha)
        return logits
    
    def run_E(self, img1, img2, scale=None, blur_sigma=1):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img1.device).div(blur_sigma).square().neg().exp2()
                img1 = upfirdn2d.filter2d(img1, f / f.sum())

        logits = self.D.pair_disc(img1, img2, scale=scale)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        loss_Gpair = 0
        loss_Dpair_gen = 0
        loss_Dpair_real = 0

        loss_Gmain = 0
        loss_Dreal = 0
        loss_Dgen = 0
        loss_rec = 0

        warmup = min(max(min((cur_nimg - self.warmup_nimg) / (self.warmup_nimg * 4), 1), 0.2), 0.9)
        # warmup = 1

        # real_img_low = torch.nn.functional.interpolate(real_img, size=(32, 32), mode='bilinear')
        real_img_high = real_img
        
        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img_high, gen_img_low, scale, h_proj, enc, alpha = self.run_G(real_img, gen_z, gen_c, temp_max=warmup)
                gen_logits = self.run_D(gen_img_high, h_proj, gen_c, scale, alpha, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()
                # loss_rec = (h_proj - enc.detach()).square().mean() * 0
                # loss_rec = -((enc @ enc.T) / enc.shape[-1]).log_softmax(dim=-1).diag().mean()
                # loss_rec = (self.G_ema.synthesis.h_proj(enc) - self.G.synthesis.h_proj(h_proj)).square().mean()
                # gen_pair_logits = self.run_E(gen_img_low, h_proj, scale)
                # loss_Gpair = (F.relu(torch.ones_like(gen_pair_logits) * (scale) - gen_pair_logits)).mean()
                # loss_Gpair = (-gen_pair_logits).mean() * warmup

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Loss/G/pair_loss', loss_Gpair)
                training_stats.report('Loss/G/loss_rec', loss_rec)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gpair + loss_rec).backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_high, gen_img_low, scale, h_proj, enc, alpha = self.run_G(real_img, gen_z, gen_c, update_emas=True, temp_max=warmup)
                gen_logits = self.run_D(gen_img_high, h_proj, gen_c, scale, alpha, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()
                # gen_pair_logits = self.run_E(gen_img_low, h_proj, scale)
                # loss_Dpair_gen = (F.relu(torch.ones_like(gen_pair_logits) * (scale) + gen_pair_logits)).mean()
                # loss_Dpair_gen = (F.relu(torch.ones_like(gen_pair_logits) + gen_pair_logits)).mean() * warmup

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + loss_Dpair_gen).backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_high_tmp = real_img_high.detach().requires_grad_(False)
                # real_img_low_tmp = real_img_low.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_high_tmp, h_proj, real_c, scale, alpha, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
                # real_pair_logits = self.run_E(real_img_low_tmp, h_proj, scale)
                # loss_Dpair_real = (F.relu(torch.ones_like(real_pair_logits) * (scale) - real_pair_logits)).mean()
                # loss_Dpair_real = (F.relu(torch.ones_like(real_pair_logits) - real_pair_logits)).mean() * warmup

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                training_stats.report('Loss/D/pair_loss', loss_Dpair_gen + loss_Dpair_real)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                (loss_Dreal + loss_Dpair_real).backward()

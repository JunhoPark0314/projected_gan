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


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    def run_G(self, real_img, z, c, update_emas=False):
        h, ws = self.G.mapping(real_img, z, c, update_emas=update_emas)
        img = self.G.synthesis(h, ws, c, update_emas=False)
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
        self.buffer = {0:None, 1:None, 2:None}

    def run_G(self, real_img, z, c, update_emas=False):
        h, ws, temb = self.G.mapping(real_img, z, c, update_emas=update_emas)
        gen = self.G.synthesis(h, ws, c, temb, update_emas=False)
        return gen

    def run_D(self, high, low, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(high, low, c)
        return logits
    
    def run_E(self, img1, img2):
        logits = self.D.pair_disc(img1, img2)
        return logits
    
    def append_buffer(self, real_img):
        self.input_condition[0] = torch.randn_like(real_img).cpu()
        self.target_real[0] = real_img.cpu()

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        self.append_buffer(real_img)

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        loss_Gpair = 0
        loss_Dpair_gen = 0
        loss_Dpair_real = 0
        warmup = max(min((cur_nimg - self.warmup_nimg) / self.warmup_nimg, 1), 0.001)
        input_condition = []
        target_real = []
        timestep = []
        for i in range(3):
            if self.input_condition[i] is not None:
                input_condition.append(self.input_condition[i])
                target_real.append(self.target_real[i])
                timestep.append(torch.ones(len(self.input_condition[i])) * (i + 1))

        input_condition = torch.cat(input_condition).to(real_img.device)
        target_real = torch.cat(target_real).to(real_img.device)
        timestep = torch.cat(timestep).to(real_img.device)

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(input_condition, gen_z, gen_c, timestep)
                gen_logits = self.run_D(gen_img, target_real, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gpair).backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(input_condition, gen_z, gen_c, timestep, update_emas=True)
                gen_logits = self.run_D(gen_img, target_real, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) * timestep * 0.5 + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + loss_Dpair_gen).backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                target_real_tmp = target_real.detach().requires_grad_(False)
                real_logits = self.run_D(target_real_tmp, target_real_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) * timestep * 0.5 - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                training_stats.report('Loss/D/pair_loss', loss_Dpair_gen + loss_Dpair_real)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                (loss_Dreal + loss_Dpair_real).backward()
        
            # update buffer on Discriminator update
            target_1 = timestep.long() == 0
            target_2 = timestep.long() == 1
            self.input_condition[1] = gen_img[target_1].detach().cpu()
            self.input_condition[2] = gen_img[target_2].detach().cpu()
            self.target_real[1] = target_real[target_1].detach().cpu()
            self.target_real[2] = target_real[target_2].detach().cpu()
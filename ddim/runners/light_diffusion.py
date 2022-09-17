import numpy as np
import torch
from ddim.functions.denoising import compute_alpha
import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        
        self.min_num_step = args.num_step
        self.num_step = args.num_step#int(self.num_timesteps * 0.7)

    # def sample_noised_pair(self, x, next_t, prev_t, netTrg):
    #     e1 = torch.randn_like(x)
    #     e2 = torch.randn_like(x)
    #     a_next = compute_alpha(self.betas, next_t)
    #     a_prev = compute_alpha(self.betas, prev_t)

    #     noised_next = a_next.sqrt() * x + (1-a_next).sqrt() * (e1)
    #     a_ts = (a_prev / a_next)
    #     var_ts = ((1-a_prev) - (1-a_next) * a_ts)
    #     noised_prev = a_ts.sqrt() * noised_next + var_ts.sqrt() * e2

    #     xt = torch.cat([noised_prev, noised_next])
    #     timestep = torch.cat([prev_t, next_t])
    #     a = torch.cat([a_prev, a_next])
    #     eps = netTrg(xt, timestep)

    #     x0 = (xt - (1 - a).sqrt() * eps) / a.sqrt()
    #     prev_x0, next_x0 = torch.chunk(x0, 2, 0)
    #     return a_next, a_prev, noised_next, noised_prev, prev_x0

    def sample_noised_pair(self, x, next_t, prev_t, netTrg):
        e1 = torch.randn_like(x)
        e2 = torch.randn_like(x)
        a_next = compute_alpha(self.betas, next_t)
        a_prev = compute_alpha(self.betas, prev_t)

        noised_next = a_next.sqrt() * x + (1-a_next).sqrt() * (e1)
        a_ts = (a_prev / a_next)
        var_ts = ((1-a_prev) - (1-a_next) * a_ts)
        noised_prev = a_ts.sqrt() * noised_next + var_ts.sqrt() * e2

        xt = torch.cat([noised_prev, noised_next])
        timestep = torch.cat([prev_t, next_t])
        a = torch.cat([a_prev, a_next])
        eps = netTrg(xt, timestep)

        # xt = noised_prev
        # timestep = prev_t
        # a = a_prev
        # eps = netTrg(xt, timestep)

        x0 = (xt - (1 - a).sqrt() * eps) / a.sqrt()
        # prev_x0 = x0
        prev_x0, next_x0 = torch.chunk(x0, 2, 0)
        return a_next, a_prev, noised_next, noised_prev, prev_x0, next_x0
    
    def sample_noised(self, x, t):
        noise = torch.randn_like(x)
        a = compute_alpha(self.betas, t)
        return a.sqrt() * x + (1-a).sqrt() * noise

    def denoise_step(self, xt, eps, t, **kwargs):
        prev_t = torch.clip(t, min=0, max=self.num_timesteps)
        a_prev = compute_alpha(self.betas, prev_t.long())
        x0_t = (xt - eps * (1 - a_prev).sqrt()) / (a_prev.sqrt())

        return x0_t
    
    def sample_image(self, model, netTrg, x_init, z=None):
        with torch.no_grad():
            step_size = self.num_timesteps // self.num_step
            n = x_init.size(0)
            t = torch.ones(n).to(x_init.device) * (self.num_timesteps - step_size//2)
            etrg = netTrg(x_init, t)
            at = compute_alpha(self.betas, t.long())
            x_init = (x_init - etrg * (1 - at).sqrt()) / at.sqrt()
            assert z.shape[1] == (self.num_step + 1)
            x0_list = [x_init.cpu()[None,...]]
            x_t = x_init
            for i in range(self.num_step + 1):
                # x_init = model(x_init, t, z[:,i])
                eps_pred = model(x_t, t, z[:,i])
                at = compute_alpha(self.betas, t.long())
                at_next = compute_alpha(self.betas, (t - step_size).clip(min=0, max=self.num_timesteps).long())
                x_s = self.gen_noised(eps_pred, at, at_next, x_t)
                x0_list.append(x_s.cpu()[None,...])
                x_t = x_s
                t = torch.clip(t-step_size, min=0, max=self.num_timesteps)
            return torch.cat(x0_list)

    def gen_noised(self, et, at, at_next, xt, **kwargs):
        # alpha_ts = (alpha_t / alpha_s)
        # var_s = (1 - alpha_s)
        # var_t = (1 - alpha_t)
        # var_ts = var_t - alpha_ts * var_s
        # zt_coeff = alpha_ts.sqrt() * var_s / var_t
        # x0_coeff = alpha_s.sqrt() * var_ts / var_t
        # return zt_coeff * xt + x0_coeff * x0_t, None
        return et

        # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        # c1 = (
        #     kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        # )
        # c2 = ((1 - at_next) - c1 ** 2).sqrt()
        # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et
        # if kwargs.get("return_x0", False):
        #     return xt_next, x0_t
        # return xt_next
    
    def sample_time(self, x):
        step_size = self.num_timesteps // self.num_step
        prev_t = torch.randint(0, self.num_timesteps, (x.size(0),), device=x.device)
        next_t = prev_t - step_size

        prev_t = torch.clip(prev_t, min=0, max=self.num_timesteps).long()
        next_t = torch.clip(next_t, min=0, max=self.num_timesteps).long()

        return next_t, prev_t
    
    def update_num_step(self):
        self.num_step = max(self.min_num_step, int(self.num_step*0.7))
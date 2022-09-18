# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F

# To guarantee same augmentation to pair

def DiffAugment_pair(x1, x2, policy='', channels_first=True):
    # x1, x2 should have same shape
    assert x1.shape == x2.shape
    if policy:
        if not channels_first:
            x1 = x1.permute(0, 3, 1, 2)
            x2 = x2.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x1, x2 = f(x1, x2)
        if not channels_first:
            x1 = x1.permute(0, 3, 1, 2)
            x2 = x2.permute(0, 3, 1, 2)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
    return x1, x2


def rand_brightness(x1, x2):
    br = (torch.rand(x1.size(0), 1, 1, 1, dtype=x1.dtype, device=x1.device) - 0.5)
    x1 = x1 + br
    x2 = x2 + br
    return x1, x2


def rand_saturation(x1, x2):
    sat = (torch.rand(x1.size(0), 1, 1, 1, dtype=x1.dtype, device=x1.device) * 2)
    x1_mean = x1.mean(dim=1, keepdim=True)
    x2_mean = x2.mean(dim=1, keepdim=True)

    x1 = (x1 - x1_mean) * sat  + x1_mean
    x2 = (x2 - x2_mean) * sat  + x1_mean

    return x1, x2


def rand_contrast(x1, x2):
    cont = (torch.rand(x1.size(0), 1, 1, 1, dtype=x1.dtype, device=x1.device) + 0.5)
    x1_mean = x1.mean(dim=[1, 2, 3], keepdim=True)
    x2_mean = x2.mean(dim=[1, 2, 3], keepdim=True)
    x1 = (x1 - x1_mean) * cont + x1_mean
    x2 = (x2 - x1_mean) * cont + x2_mean
    return x1, x2


def rand_translation(x1, x2, ratio=0.125):
    x = x1
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)


    x1_pad = F.pad(x1, [1, 1, 1, 1, 0, 0, 0, 0])
    x1 = x1_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    x2_pad = F.pad(x2, [1, 1, 1, 1, 0, 0, 0, 0])
    x2 = x2_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x1, x2


def rand_cutout(x1, x2, ratio=0.2):
    x = x1
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x1 = x1 * mask.unsqueeze(1)
    x2 = x2 * mask.unsqueeze(1)
    return x1, x2

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

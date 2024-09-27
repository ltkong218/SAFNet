import math
import numpy as np
import torch
import torch.nn.functional as F


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def weight_3expo_low_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.0
    mask2 = img >= 0.50
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w

def weight_3expo_mid_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w

def weight_3expo_high_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 0.0
    w /= 0.5
    return w

def merge_hdr(ldr_imgs, lin_imgs, mask0, mask2):
    sum_img = torch.zeros_like(ldr_imgs[1])
    sum_w = torch.zeros_like(ldr_imgs[1])
    w_low = weight_3expo_low_tog17(ldr_imgs[1]) * mask0
    w_mid = weight_3expo_mid_tog17(ldr_imgs[1]) + weight_3expo_low_tog17(ldr_imgs[1]) * (1.0 - mask0) + weight_3expo_high_tog17(ldr_imgs[1]) * (1.0 - mask2)
    w_high = weight_3expo_high_tog17(ldr_imgs[1]) * mask2
    w_list = [w_low, w_mid, w_high]
    for i in range(len(ldr_imgs)):
        sum_w += w_list[i]
        sum_img += w_list[i] * lin_imgs[i]
    hdr_img = sum_img / (sum_w + 1e-9)
    return hdr_img

def range_compressor(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)

def calculate_psnr(img1, img2):
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr

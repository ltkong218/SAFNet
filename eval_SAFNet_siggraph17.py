import os
import shutil
import time
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SAFNet import SAFNet
from datasets import SIGGRAPH17_Test_Dataset
from utils import range_compressor, calculate_psnr


dataset_eval = SIGGRAPH17_Test_Dataset(dataset_dir='/Datasets/SIGGRAPH17_Dataset/Test')

model = SAFNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/SAFNet_siggraph17.pth'))

test_results_path = './img_hdr_pred_siggraph17'

if os.path.exists(test_results_path):
    shutil.rmtree(test_results_path)
    
if not os.path.exists(test_results_path):
    os.makedirs(test_results_path)

psnr_l_list = []
psnr_m_list = []

for i in range(len(dataset_eval)):
    imgs_lin, imgs_ldr, expos, img_hdr_gt = dataset_eval[i]
    img0_c = torch.cat([torch.from_numpy(imgs_lin[0]), torch.from_numpy(imgs_ldr[0])], 2).permute(2, 0, 1).unsqueeze(0).cuda()
    img1_c = torch.cat([torch.from_numpy(imgs_lin[1]), torch.from_numpy(imgs_ldr[1])], 2).permute(2, 0, 1).unsqueeze(0).cuda()
    img2_c = torch.cat([torch.from_numpy(imgs_lin[2]), torch.from_numpy(imgs_ldr[2])], 2).permute(2, 0, 1).unsqueeze(0).cuda()
    img_hdr_gt = torch.from_numpy(img_hdr_gt).permute(2, 0, 1).unsqueeze(0).cuda()

    with torch.no_grad():
        img_hdr_m, img_hdr_r = model(img0_c, img1_c, img2_c)

    psnr_l = calculate_psnr(img_hdr_r, img_hdr_gt).cpu()

    img_hdr_r_m = range_compressor(img_hdr_r)
    img_hdr_gt_m = range_compressor(img_hdr_gt)
    psnr_m = calculate_psnr(img_hdr_r_m, img_hdr_gt_m).cpu()

    psnr_l_list.append(psnr_l)
    psnr_m_list.append(psnr_m)

    print('SIGGRAPH17 Test {:03d}: PSNR_l={:.3f} PSNR_m={:.3f}'.format(i+1, psnr_l, psnr_m))

    img_hdr_r_np = img_hdr_r[0].data.cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(test_results_path, '{:03d}.hdr'.format(i+1)), img_hdr_r_np[:, :, ::-1])

print('SIGGRAPH17 Test Average: PSNR_l={:.3f} PSNR_m={:.3f}'.format(np.array(psnr_l_list).mean(), np.array(psnr_m_list).mean()))

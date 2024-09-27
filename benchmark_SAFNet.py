import os
import time
import numpy as np
import torch
from models.SAFNet import SAFNet
import pynvml
from thop import profile


model = SAFNet().cuda().eval()

img0_c = torch.randn(1, 6, 1000, 1500).cuda()
img1_c = torch.randn(1, 6, 1000, 1500).cuda()
img2_c = torch.randn(1, 6, 1000, 1500).cuda()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpuName = pynvml.nvmlDeviceGetName(handle)
print(gpuName)

with torch.no_grad():
    for i in range(10):
        out = model(img0_c, img1_c, img2_c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(100):
        out = model(img0_c, img1_c, img2_c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('Time: {:.3f}s'.format((time.time() - time_stamp) / 100))

flops, params = profile(model, inputs=(img0_c, img1_c, img2_c, 0.5, True), verbose=False)
print('FLOPs: {:.3f}T, Params: {:.2f}M'.format(flops / 1000 / 1000 / 1000 / 1000, params / 1000 / 1000))

# SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging
The official PyTorch implementation of SAFNet (ECCV 2024).

Authors: [Lingtong Kong](https://scholar.google.com.hk/citations?user=KKzKc_8AAAAJ&hl=zh-CN), Bo Li, Yike Xiong, Hao Zhang, Hong Gu, Jinwei Chen


## Abstract
Reconstructing High Dynamic Range (HDR) image from multiple Low Dynamic Range (LDR) images with different exposures is a challenging task when facing truncated texture and complex motion. Existing deep learning-based methods have achieved great success by either following the alignment and fusion pipeline or utilizing attention mechanism. However, the large computation cost and inference delay hinder them from deploying on resource limited devices. In this paper, to achieve better efficiency, a novel Selective Alignment Fusion Network (SAFNet) for HDR imaging is proposed. After extracting pyramid features, it jointly refines valuable area masks and cross-exposure motion in selected regions with shared decoders, and then fuses high quality HDR image in an explicit way. This approach can focus the model on finding valuable regions while estimating their easily detectable and meaningful motion. For further detail enhancement, a lightweight refine module is introduced which enjoys privileges from previous optical flow, selection masks and initial prediction. Moreover, to facilitate learning on samples with large motion, a new window partition cropping method is presented during training. Experiments on public and newly developed challenging datasets show that proposed SAFNet not only exceeds previous SOTA competitors quantitatively and qualitatively, but also runs order of magnitude faster. Code and dataset will be available soon.

## Overall Performance

![](./data/fig1.PNG)

## Quantitative Results

![](./data/fig2.PNG)

## Qualitative Results

![](./data/fig3.PNG)

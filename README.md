# SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging
The official PyTorch implementation of [SAFNet](https://arxiv.org/abs/2407.16308) (ECCV 2024). The technology and codes related to this paper are for academic research only and no commercial use for any purpose is allowed.

Authors: [Lingtong Kong](https://scholar.google.com.hk/citations?user=KKzKc_8AAAAJ&hl=zh-CN), Bo Li, Yike Xiong, Hao Zhang, Hong Gu, Jinwei Chen


## Abstract
Multi-exposure High Dynamic Range (HDR) imaging is a challenging task when facing truncated texture and complex motion. Existing deep learning-based methods have achieved great success by either following the alignment and fusion pipeline or utilizing attention mechanism. However, the large computation cost and inference delay hinder them from deploying on resource limited devices. In this paper, to achieve better efficiency, a novel Selective Alignment Fusion Network (SAFNet) for HDR imaging is proposed. After extracting pyramid features, it jointly refines valuable area masks and cross-exposure motion in selected regions with shared decoders, and then fuses high quality HDR image in an explicit way. This approach can focus the model on finding valuable regions while estimating their easily detectable and meaningful motion. For further detail enhancement, a lightweight refine module is introduced which enjoys privileges from previous optical flow, selection masks and initial prediction. Moreover, to facilitate learning on samples with large motion, a new window partition cropping method is presented during training. Experiments on public and newly developed challenging datasets show that proposed SAFNet not only exceeds previous SOTA competitors quantitatively and qualitatively, but also runs order of magnitude faster.

## Overall Performance

![](./data/fig1.PNG)

## Quantitative Results

![](./data/fig2.PNG)

## Qualitative Results

![](./data/fig3.PNG)

## Challenge123 Dataset
The existing labeled multi-exposure HDR datasets have facilitated research in related fields. However, results of recent methods tend to be saturated due to their limited evaluative ability. We attribute this phenomenon to most of their samples having relatively small motion magnitude between LDR inputs and relatively small saturation ratio of the reference image. To probe the performance gap between different algorithms, we propose a new challenging multi-exposure HDR dataset with enhanced motion range and saturated regions. There are 96 training samples and 27 test samples in our developed Challenge123 dataset.

Dataset download link: [https://huggingface.co/datasets/ltkong218/Challenge123](https://huggingface.co/datasets/ltkong218/Challenge123).

To enhance the applicability of our dataset and promote future research, for each of three content-related moving scenes, we further create under-, middle- and over-exposure LDR images and corresponding HDR image. It means that for each of our 96 training scenes, we have $3 \times 2 \times 1 = 6$ exposure combination for training theoretically, while all experiments on our Challenge123 dataset in this paper adopt under-, middle- and over-exposure LDR images by the time order like previous methods.

Training samples for our experiments in this paper:
<pre><code>./Training/xxx_1/ldr_img_1.tif</code>
<code>./Training/xxx_2/ldr_img_2.tif</code>
<code>./Training/xxx_3/ldr_img_3.tif</code>
<code>./Training/xxx_2/exposure.txt</code>
<code>./Training/xxx_2/hdr_img.hdr</code></pre>

Test samples for our experiments in this paper:
<pre><code>./Test/xxx_1/ldr_img_1.tif</code>
<code>./Test/xxx_2/ldr_img_2.tif</code>
<code>./Test/xxx_3/ldr_img_3.tif</code>
<code>./Test/xxx_2/exposure.txt</code>
<code>./Test/xxx_2/hdr_img.hdr</code></pre>

<code>xxx</code> means the three digits data ID.

## Training
I am sorry that I can not release the training code of SAFNet due to requirements of my company, but the readers can try to reproduce the experimental results according to my paper.

## Evaluation on Kalantari 17 Test Dataset
To test PSNR-m and PSNR-l, set the right dataset path in <code>eval_SAFNet_siggraph17.py</code> and <code>eval_SAFNet_S_siggraph17.py</code>, and then run
<pre><code>$ python eval_SAFNet_siggraph17.py</code>
<code>$ python eval_SAFNet_S_siggraph17.py</code></pre>
To test SSIM-m, SSIM-l and HDR-VDP2, (1) get predicted HDR images in folder <code>./img_hdr_pred_siggraph17</code> by running <code>$ python eval_SAFNet_siggraph17.py</code>; (2) put and rename the ground truth HDR test images into folder <code>./matlab_evaluation/img_hdr_gt_siggraph17/*</code> as <code>001.hdr, 002.hdr, ...</code>; (3) download file [hdrvdp-2.2.2](https://sourceforge.net/projects/hdrvdp/files/hdrvdp/2.2.2/hdrvdp-2.2.2.zip/download) and put the unzipped file into folder <code>./matlab_evaluation/hdrvdp-2.2.2</code>. Run matlab script
<pre><code>./matlab_evaluation/eval_siggraph17.m</code></pre>

## Evaluation on Challenge123 Test Dataset
To test PSNR-m and PSNR-l, set the right dataset path in <code>eval_SAFNet_challenge123.py</code>, and then run
<pre><code>$ python eval_SAFNet_challenge123.py</code></pre>
To test SSIM-m, SSIM-l and HDR-VDP2, (1) get predicted HDR images in folder <code>./img_hdr_pred_challenge123</code> by running <code>$ python eval_SAFNet_challenge123.py</code>; (2) put and rename the ground truth HDR test images into folder <code>./matlab_evaluation/img_hdr_gt_challenge123/*</code> as <code>001.hdr, 002.hdr, ...</code>; (3) download file [hdrvdp-2.2.2](https://sourceforge.net/projects/hdrvdp/files/hdrvdp/2.2.2/hdrvdp-2.2.2.zip/download) and put the unzipped file into folder <code>./matlab_evaluation/hdrvdp-2.2.2</code>. Run matlab script
<pre><code>./matlab_evaluation/eval_challenge123.m</code></pre>

## Benchmarks for SAFNet and SAFNet-S
To test running time, model parameters and computation complexity (FLOPs), you can run
<pre><code>$ python benchmark_SAFNet.py</code>
<code>$ python benchmark_SAFNet_S.py</code></pre>
Before, you may run <code>pip install thop</code> and <code>pip install pynvml</code>.

## Citation
When using any parts of the Dataset, Software or the Paper in your work, please cite the following paper:
<pre><code>@InProceedings{Kong_2024_ECCV, 
  author={Kong, Lingtong and Li, Bo and Xiong, Yike and Zhang, Hao and Gu, Hong and Chen, Jinwei}, 
  title={SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging}, 
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)}, 
  year={2024}
}</code></pre>

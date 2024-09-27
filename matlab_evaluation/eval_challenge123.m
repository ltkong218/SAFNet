addpath(genpath('hdrvdp-2.2.2'));

hdr_gt_dir = './img_hdr_gt_challenge123';
hdr_pred_dir = '../img_hdr_pred_challenge123';

hdrs = dir(fullfile(hdr_gt_dir, '*.hdr'));
num = numel(hdrs);
psnr_l_all = zeros(num, 1);
ssim_l_all = zeros(num, 1);
psnr_m_all = zeros(num, 1);
ssim_m_all = zeros(num, 1);
hdrvdp_all = zeros(num, 1);

for i = 1:num
    gt_hdr = hdrread(fullfile(hdr_gt_dir, hdrs(i).name));
    pred_hdr = hdrread(fullfile(hdr_pred_dir, hdrs(i).name));
    gt_hdr_m = mulog_tonemap(gt_hdr, 5000);
    pred_hdr_m = mulog_tonemap(pred_hdr, 5000);
    
    psnr_l = psnr(pred_hdr, gt_hdr);
    psnr_l_all(i) = psnr_l;
    ssim_l = ssim(pred_hdr, gt_hdr);
    ssim_l_all(i) = ssim_l;
    
    psnr_m = psnr(pred_hdr_m, gt_hdr_m);
    psnr_m_all(i) = psnr_m;
    ssim_m = ssim(pred_hdr_m, gt_hdr_m);
    ssim_m_all(i) = ssim_m;

    ppd = hdrvdp_pix_per_deg(28, [size(gt_hdr,2) size(gt_hdr, 1)], 0.55);
    hdrvdp_res = hdrvdp(pred_hdr, gt_hdr, 'sRGB-display', ppd);
    hdrvdp_all(i) = hdrvdp_res.Q;
    fprintf('%02d/%02d: PSNR_l %.3f  SSIM_l %.4f  PSNR_m %.3f  SSIM_m %.4f  HDR-VDP %.2f\n', i, num, psnr_l_all(i), ssim_l_all(i), psnr_m_all(i), ssim_m_all(i), hdrvdp_all(i));
end

fprintf('Challenge123 Test Average PSNR_l %.3f  SSIM_l %.4f  PSNR_m %.3f  SSIM_m %.4f  HDR-VDP %.2f\n', mean(psnr_l_all), mean(ssim_l_all), mean(psnr_m_all), mean(ssim_m_all), mean(hdrvdp_all));

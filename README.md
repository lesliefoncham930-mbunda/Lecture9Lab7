# Lecture9Lab7
Lab 7: Gaussian Noise Denoising – Traditional vs AI (DnCNN)
1) Setup: Gaussian noise
cameraman.tif: 256×256 grayscale, converted to double in [0,1].
imnoise(…,'gaussian',0,σ²) with σ = 0.04 (variance 0.0016). This adds zero-mean additive white Gaussian noise (AWGN) to every pixel: I_noisy = I + n, with n ~ N(0, σ²). It models sensor and electronic noise.
In the figure, the noisy image has fine grain spread evenly across the sky, the coat and the grass.
2) Traditional filter: adaptive Wiener (wiener2, 5×5)

The Wiener filter is adaptive. In each 5×5 window it estimates the local mean μ and variance σ_L², then computes:

out = μ + (max(σ_L² − ν², 0) / σ_L²) · (I − μ)

where ν² = 0.0016 is the known noise variance.

Flat areas (sky, ground): σ_L² ≈ ν², so the output is close to μ and is fully smoothed.
Edges and detail (camera, tripod, coat outline): σ_L² ≫ ν², so the output stays close to the original pixel and the edge is kept.

It is the optimal linear filter in the mean-square-error sense for AWGN.

3) AI filter: DnCNN

DnCNN (Denoising Convolutional Neural Network) is a deep CNN of about 20 layers (convolution + batch normalization + ReLU). It uses residual learning: the network predicts the noise n̂, and the clean image is I_noisy − n̂.

It was trained on many image patches with Gaussian noise added, so it learned what natural textures and edges look like compared with noise.
Unlike a fixed filter, it applies learned, non-linear filters across many layers, so it can tell fine texture from noise better.
4) Quantitative results
Method	MSE	PSNR (dB)	SSIM
Noisy image	0.00154	28.13	0.631
Traditional (Wiener 5×5)	0.00061	32.13	0.882
AI (DnCNN)	0.00056	32.51	0.896

MSE values are derived from PSNR: MSE = 10^(−PSNR/10), since the peak value is 1. Use the MSE values from the extra code below once you run it.

What the metrics mean:

MSE: average squared pixel error. Lower is better. The noisy MSE (0.00154) is close to σ² = 0.0016, as expected; it is slightly lower because values are clipped to [0,1].
PSNR = 10·log₁₀(1/MSE): error on a log scale relative to the peak value. Higher is better. +3 dB ≈ half the MSE.
SSIM: compares local luminance, contrast and structure between the two images, on a scale from 0 to 1. Higher is better. It matches human perception of edges and texture more closely than MSE does.

What the results show:

Both filters help a lot. Wiener improves PSNR by +4.0 dB (MSE reduced by about 60%) and SSIM from 0.63 to 0.88.
DnCNN is best on both metrics, but the margin over Wiener is small: +0.37 dB PSNR and +0.013 SSIM.
The noise level is low (σ = 0.04 ≈ 10/255), so a well-tuned Wiener filter, given the true noise variance, already does well. The advantage of DnCNN typically grows at higher noise levels.
5) Visual observations
Noisy: visible grain everywhere, especially in the smooth sky and the dark coat.
Wiener: the sky and ground are smooth and the main edges (camera, tripod, figure outline) are kept. Slight blotchy texture and a little softening remain on fine detail such as the grass and the buildings in the background.
DnCNN: the cleanest result, with smooth flat areas, sharp edges, and fine detail kept slightly better. At this display size it looks very close to the Wiener result, which agrees with the small difference in the metrics.

Tip for your screenshot: the subplot titles are cut off (“Original Im…”, “Noisy Ima…”). Make the figure wider before saving it, for example figure('Name','…','Position',[100 100 1400 400]);, or use shorter titles.

Conclusion and limitations
SSIM vs PSNR: SSIM improves much more than the PSNR numbers suggest (0.63 → 0.90). Denoising restores structure, not just pixel accuracy.
Wiener is fast, needs no training, and is easy to explain, but it assumes the noise is stationary Gaussian, needs the noise variance (provided here), and uses a fixed window, so it slightly smooths fine texture.
DnCNN gives the best quality and works without being told σ, but it needs a pre-trained network (the Deep Learning Toolbox), more computation, and it is a “black box”. It is also only reliable for the noise type and range it was trained on. On salt & pepper noise, for example, it would do much worse than a median filter.
Extra code: MSE + mean, median and Gaussian blur (covers the learning outcomes)

Add this after Section 4:

matlab
%% 6) Classic filters + full metric table (MSE, PSNR, SSIM)
meanFiltered   = imfilter(noisyImage, fspecial('average',3), 'replicate');
medianFiltered = medfilt2(noisyImage, [3 3]);
gaussFiltered  = imgaussfilt(noisyImage, 0.8);

imgs  = {noisyImage, meanFiltered, medianFiltered, gaussFiltered, traditionalFiltered, aiFiltered};
names = {'Noisy','Mean 3x3','Median 3x3','Gaussian blur','Wiener 5x5','DnCNN'};

fprintf('\n%-14s | %8s | %9s | %6s\n','Method','MSE','PSNR (dB)','SSIM');
fprintf('%s\n', repmat('-',1,46));
for k = 1:numel(imgs)
    fprintf('%-14s | %8.5f | %9.2f | %6.4f\n', names{k}, ...
        immse(imgs{k}, originalImage), psnr(imgs{k}, originalImage), ssim(imgs{k}, originalImage));
end

figure('Position',[100 100 1400 450]);
montage({originalImage, noisyImage, meanFiltered, medianFiltered, ...
         gaussFiltered, traditionalFiltered, aiFiltered}, 'Size',[1 7]);
title('Original | Noisy | Mean | Median | Gaussian | Wiener | DnCNN');

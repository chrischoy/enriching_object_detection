function [imgUIdx, imgVIdx] = eod_hog_to_img_fft(hogUIdx, hogVIdx, templateSize, sbin, scale)

% convolutionMode = 1, convolution using FFT, the size of the convolution
% is size of the image + size of template -1

% padding size is templateSize(1) - 1
% (hogUIdx - (templateSize(1) - 1) - 1) = (hogUIdx - templateSize(1))  zero-base indexing
imgUIdx = (hogUIdx - templateSize(1)) * sbin/scale + 1; % zero-base indexing to resize and add 1
imgVIdx = (hogVIdx - templateSize(2)) * sbin/scale + 1; % zero-base indexing to resize and add 1 

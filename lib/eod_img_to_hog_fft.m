function [hogUIdx, hogVIdx] = eod_img_to_hog_fft(imgUIdx, imgVIdx, sbin, scale)

% padding is (templateSize(2) - 1) long and (templateSize(1) - 1) high
% for HeatMap but not for hog. HOG don't have any padding
hogUIdx = ( ( imgUIdx - 1 ) * scale / sbin ) + 1;
hogVIdx = ( ( imgVIdx - 1 ) * scale / sbin ) + 1;

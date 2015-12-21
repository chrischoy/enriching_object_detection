function [hogUIdx, hogVIdx] = eod_img_to_hog_conv(imgUIdx, imgVIdx, sbin, scale, padder)

hogUIdx = ( imgUIdx - 1 ) * scale / sbin + padder + 1;
hogVIdx = ( imgVIdx - 1 ) * scale / sbin + padder + 1;

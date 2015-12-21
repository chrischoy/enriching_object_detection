function [feat] = dwot_hog(im, scale, params)

sbin = params.sbin;
scaled = resizeMex(im,scale);
feat = params.init_params.features(scaled,sbin);

feat = padarray(feat, [1 1 0], 0);

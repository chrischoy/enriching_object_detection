function [bbsAllLevel, hog, scales] = eod_detect(I, templates, param)

doubleIm = im2double(I);
[hog, scales] = esvm_pyramid(doubleIm, param);
hogPadder = param.detect_pyramid_padding;
sbin = param.sbin;

nTemplates =  numel(templates);
sz = cellfun(@(x) size(x), templates, 'UniformOutput',false);

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
hog = hog(minsizes >= param.min_hog_length);
scales = scales(minsizes >= param.min_hog_length);
bbsAll = cell(length(hog),1);

for level = length(hog):-1:1
  hog{level} = padarray(single(hog{level}), [hogPadder hogPadder 0], 0); % Convolution, same size
  HM = fconvblasfloat(hog{level}, templates, 1, nTemplates);

  rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
  scale = scales(level);
  templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
  bbsTemplate = cell(nTemplates,1);
  
  for templateIdx = templateIdxes    
    [idx] = find(HM{templateIdx}(:) > param.detection_threshold);
    if isempty(idx)
      continue;
    end

    [y_coord,x_coord] = ind2sub(rmsizes{templateIdx}(1:2), idx);

    % HOG templates are consistently smaller. Add extra padding after
    % detection
    [y1, x1] = dwot_hog_to_img_conv(y_coord - 0.5, x_coord - 0.5, sbin, scale, hogPadder);
    [y2, x2] = dwot_hog_to_img_conv(y_coord + sz{templateIdx}(1) + 0.5, x_coord + sz{templateIdx}(2) + 0.5, sbin, scale, hogPadder);
    
    bbs = zeros(numel(y_coord), 12);
    bbs(:,1:4) = [x1 y1, x2, y2];

    bbs(:,5) = scale;
    bbs(:,6) = level;
    bbs(:,7) = y_coord;
    bbs(:,8) = x_coord;

    % bbs(:,9) is designated for overlap
    % bbs(:,10) is designated for GT index
    
    % bbs(:,9) = boxoverlap(bbs, annotation.bbox + [0 0 annotation.bbox(1:2)]);
    % bbs(:,10) = abs(detectors{templateIdx}.az - azGT) < 30;
    bbs(:,10) = param.detectors{templateIdx}.az;
    bbs(:,11) = templateIdx;
    bbs(:,12) = HM{templateIdx}(idx);

    bbsTemplate{templateIdx} = bbs;
    
    % if visualize
    if 0
      subplot(231); imagesc(HOGpicture(templates{templateIdx})); axis equal; axis tight;
      subplot(232); imagesc(param.detectors{templateIdx}.rendering_image); axis equal; axis tight; axis off;
      text(10,20,{['score ' num2str(bbs(Idx,12))],['azimuth ' num2str(bbs(Idx,10))]},'BackgroundColor',[.7 .9 .7]);
      subplot(233); imagesc(HM{templateIdx}); %caxis([100 200]); 
      colorbar; axis equal; axis tight; 
      subplot(234); imagesc(doubleIm); axis equal; axis tight; axis off;
      rectangle('Position',bbs(Idx,1:4)-[0 0 bbs(Idx,1:2)]);
      drawnow;
    end
  end
  bbsAll{level} = cell2mat(bbsTemplate);
end

bbsAllLevel = cell2mat(bbsAll);

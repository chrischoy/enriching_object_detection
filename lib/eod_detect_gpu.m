function [bbsAllLevel, hog, scales] = eod_detect_gpu(I, detectors, param, visualize)

if ~isfield(param, 'gpu_detectors')
  % This caches detectors in GPU memory
  param.gpu_detectors = eod_get_detectors_gpu(detectors);
end

if nargin < 4
  visualize = true;
end

[hog, scales] = esvm_pyramid(I, param);
sbin = param.sbin;

nTemplates =  numel(detectors);

sz = cellfun(@(x) size(x), param.gpu_detectors, 'UniformOutput',false);
maxTemplateHeight = max(cellfun(@(x) x(1), sz));
maxTemplateWidth = max(cellfun(@(x) x(2), sz));

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
hog = hog(minsizes >= param.min_hog_length);
scales = scales(minsizes >= param.min_hog_length);
bbsAll = cell(length(hog),1);

for level = length(hog):-1:1

%   fhog = cudaFFTData(single(hog{level}), maxTemplateHeight, maxTemplateWidth);
%   HM = cudaConvFFTData(fhog,templates, param.cuda_conv_n_threads);
  % The following line will center the HOG feature of the image and the
  % convolution will be simply computing convolution of whitened features
  % However, if we do not include the following line, our model becomes an
  % LDA model.
  % However, this does not make a big differenc. If we center it,
  % textureless region will have large negative norms.
  % hog{level} = bsxfun(@minus, hog{level}, muSwapDim);

  HM = cudaConvolutionFFT(single(hog{level}), maxTemplateHeight, ...
    maxTemplateWidth, param.gpu_detectors, param.cuda_conv_n_threads, param.device_id);


  rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
  scale = scales(level);
  bbsTemplate = cell(nTemplates,1);

  for templateIdx = 1:nTemplates
    [idx] = find(HM{templateIdx}(:) > param.detection_threshold);

    if isempty(idx)
      continue;
    end

    [y_coord,x_coord] = ind2sub(rmsizes{templateIdx}(1:2), idx);

    % HOG templates are consistently smaller. Add extra padding after
    % detection
    [y1, x1] = eod_hog_to_img_fft(y_coord - 0.5, x_coord - 0.5, sz{templateIdx}, sbin, scale);
    [y2, x2] = eod_hog_to_img_fft(y_coord + sz{templateIdx}(1) + 0.5 , x_coord + sz{templateIdx}(2) + 0.5, sz{templateIdx}, sbin, scale);

    bbs = zeros(numel(y_coord), 12);
    bbs(:,1:4) = [x1 y1, x2, y2];
    bbs(:,5) = scale;
    bbs(:,6) = level;
    bbs(:,7) = y_coord;
    bbs(:,8) = x_coord;

    % bbs(:,9) is designated for overlap
    % bbs(:,10) is designated for GT index / obsolete

    % bbs(:,9) = boxoverlap(bbs, annotation.bbox + [0 0 annotation.bbox(1:2)]);
    % bbs(:,10) = abs(detectors{templateIdx}.az - azGT) < 30; for 3D object
    % dataset
    bbs(:,10) = detectors{templateIdx}.az;
    bbs(:,11) = templateIdx;
    bbs(:,12) = HM{templateIdx}(idx);
    bbsTemplate{templateIdx} = bbs;

    if visualize
      [score, Idx] = max(bbs(:,12));
      subplot(221); imagesc(HOGpicture(detectors{templateIdx}.whow)); axis equal; axis tight;
      subplot(222); imagesc(detectors{templateIdx}.rendering_image); axis equal; axis tight; axis off;
      Idx = 1;
      text(10,20,{['score ' num2str(bbs(Idx,12))],['azimuth ' num2str(bbs(Idx,10))]},'BackgroundColor',[.7 .9 .7]);
      subplot(223); imagesc(HM{templateIdx}); %caxis([100 200]); 
      colorbar; axis equal; axis tight; 
      subplot(224); imagesc(I); axis equal; axis tight; axis off;
      rectangle('Position',bbs(Idx,1:4)-[0 0 bbs(Idx,1:2)]);
      
      drawnow;
    end
  end
  bbsAll{level} = cell2mat(bbsTemplate);
end

bbsAllLevel = cell2mat(bbsAll);

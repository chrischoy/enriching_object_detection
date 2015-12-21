function [hog_region_pyramid, im_region] = eod_extract_hog(hog, scales, detectors, bbs_nms, param, im, visualize)

% Clip bounding box to fit image.

% Create HOG pyramid for each of the proposal regions.
% hog_region : 
% im_region : 

% Padded Region
%  -------------------
%  | offset x, y
%  | 
%  |  ----- Actual image and hog region start
%  |  |
% To prevent unnecessary 

if nargin < 7
    visualize = false;
end

padder = param.detect_pyramid_padding;
sbin = param.sbin;

if isfield(param,'region_extraction_padding_ratio')
  region_extraction_padding_ratio = param.region_extraction_padding_ratio;
else
  region_extraction_padding_ratio = 0.1; % 10 percent
end

if isfield(param, 'region_extraction_levels')
  region_extraction_levels = param.region_extraction_levels;
else
  region_extraction_levels = 2;
end

% assume that bbsNMS is a matrix with row vectors
n_regions = size(bbs_nms, 1);

% bbsNMS = round(clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]));
% bbsNMS(:,1:4) = round(bbsNMS(:,1:4));
nHOG = numel(hog);

hog_region_pyramid = cell(1,n_regions);

for region_idx = 1:n_regions

  % For the level that the detection occured,
  img_x1 = bbs_nms(region_idx, 1); % x1
  img_x2 = bbs_nms(region_idx, 3); % y1

  img_y1 = bbs_nms(region_idx, 2); % y1
  img_y2 = bbs_nms(region_idx, 4); % y2

  detScale = bbs_nms(region_idx, 5); % scale
  detLevel = bbs_nms(region_idx, 6); % level
  detUIdx  = bbs_nms(region_idx, 7); % uus
  detVIdx  = bbs_nms(region_idx, 8); % uus
  detTemplateIdx = bbs_nms(region_idx, 11); % template Id
  detTemplateSize = detectors{detTemplateIdx}.sz;
  detScore = bbs_nms(region_idx, 12);
  
  startLevel = max(1, detLevel - region_extraction_levels);
  endLevel = min(nHOG, detLevel + region_extraction_levels);
  n_level = endLevel - startLevel + 1;
  
  hog_region_pyramid{region_idx}.image_bbox = bbs_nms(region_idx,1:4);
  hog_region_pyramid{region_idx}.template_idx = detTemplateIdx;
  hog_region_pyramid{region_idx}.template_size = detTemplateSize;
  hog_region_pyramid{region_idx}.det_score = detScore;
  hog_region_pyramid{region_idx}.models_path = {};
  hog_region_pyramid{region_idx}.models_idx = detectors{detTemplateIdx}.model_index;
%   hog_region_pyramid{region_idx}.pyramid = struct('hog_bbox',repmat({zeros(1,4)},nLevel,1),...
%                                                   'clip_hog_bbox',repmat({zeros(1,4)},nLevel,1),...
%                                                   'padded_clip_hog_bbox',repmat({zeros(1,4)},nLevel,1),...
%                                                   'padded_hog',cell(1, nLevel),...
%                                                   'level',zeros(nLevel,1),...
%                                                   'scale',zeros(nLevel,1));

  pyramidIdx = 1;
  
  % To extract pyramid, find the HOG index of image point (x1, y1) and (x2, y2)
  for level = startLevel:endLevel
    scale = scales(level);
    image_hog_size = size(hog{level});
    
    % U is row idx, V is col idx
    if param.computing_mode == 0
      [hog_y1, hog_x1] = eod_img_to_hog_conv(img_y1, img_x1, sbin, scale, padder);
      [hog_y2, hog_x2] = eod_img_to_hog_conv(img_y2, img_x2, sbin, scale, padder);
    elseif param.computing_mode == 1
      [hog_y1, hog_x1] = eod_img_to_hog_fft(img_y1, img_x1, sbin, scale);
      [hog_y2, hog_x2] = eod_img_to_hog_fft(img_y2, img_x2, sbin, scale);
    else
      error('computing mode not supported');
    end
    hog_y1 = floor(hog_y1);
    hog_x1 = floor(hog_x1);
    hog_y2 = ceil(hog_y2);
    hog_x2 = ceil(hog_x2);
    hog_y2 = hog_y2 - 1;
    hog_x2 = hog_x2 - 1;
    
    xpadding = ceil((hog_x2 - hog_x1) * region_extraction_padding_ratio);
    ypadding = ceil((hog_y2 - hog_y1) * region_extraction_padding_ratio);
    
    padded_hog_y1 = hog_y1 - ypadding;
    padded_hog_y2 = hog_y2 + ypadding;
    padded_hog_x1 = hog_x1 - xpadding;
    padded_hog_x2 = hog_x2 + xpadding;
    clip_padded_y1 = max(padded_hog_y1, 1);
    clip_padded_x1 = max(padded_hog_x1, 1);
    clip_padded_y2 = min(padded_hog_y2, image_hog_size(1));
    clip_padded_x2 = min(padded_hog_x2, image_hog_size(2));
    
    % To save space, extract regions fromt the reference HOG.
    % hog_region_pyramid{pyramidIdx}.clipHog
    hog_region_pyramid{region_idx}.pyramid(pyramidIdx).hog_bbox             = [hog_x1, hog_y1, hog_x2, hog_y2]; % x1 y1 x2 y2
    hog_region_pyramid{region_idx}.pyramid(pyramidIdx).padded_hog_bbox      = [padded_hog_x1, padded_hog_y1, padded_hog_x2, padded_hog_y2]; % x1 y1 x2 y2
    hog_region_pyramid{region_idx}.pyramid(pyramidIdx).clip_hog_bbox        = [clip_padded_x1, clip_padded_y1, clip_padded_x2, clip_padded_y2];
    % hog_region_pyramid{region_idx}.pyramid(pyramidIdx).padded_clip_hog_bbox = [clipV1 - xpadding, clipU1 - ypadding, clipV2 + xpadding, clipU2 - ypadding]; % x1 y1 x2 y2
    
    hog_region_pyramid{region_idx}.pyramid(pyramidIdx).padded_hog           = pad_hog_region(hog{level},...
                                                                                hog_x1, hog_y1, hog_x2, hog_y2,...
                                                                                clip_padded_x1, clip_padded_y1, clip_padded_x2, clip_padded_y2,...
                                                                                xpadding, ypadding,...
                                                                                param);
    % TODO
    % hog_region_pyramid{region_idx}.pyramid(pyramidIdx).image_coord          = 1;

    hog_region_pyramid{region_idx}.pyramid(pyramidIdx).level                = level;
    hog_region_pyramid{region_idx}.pyramid(pyramidIdx).scale                = scale;

    
    % debug
    if visualize
      subplot(221);
      % Confirmed correct
      if param.computing_mode == 0
        [img_y1_d, img_x1_d] = dwot_hog_to_img_conv(hog_y1, hog_x1, sbin, scale, padder);
        [img_y2_d, img_x2_d] = dwot_hog_to_img_conv(hog_y2 + 1, hog_x2 + 1, sbin, scale, padder);
      elseif param.computing_mode == 1
        [img_y1_d, img_x1_d] = dwot_hog_to_img_fft(hog_y1, hog_x1, [1 1], sbin, scale);
        [img_y2_d, img_x2_d] = dwot_hog_to_img_fft(hog_y2, hog_x2, [0 0], sbin, scale);
      else
        error('computing mode not supported');
      end
    
      imagesc(im);
      rectangle('position',[img_x1_d, img_y1_d, img_x2_d-img_x1_d, img_y2_d-img_y1_d]);
      axis equal; axis tight;
      
      subplot(222);
      hogSize = 20;
      imagesc(HOGpicture(hog{level},hogSize));
      dwot_draw_hog_bounding_box(hog_x1,        hog_y1,         hog_x2,       hog_y2,        hogSize);
      dwot_draw_hog_bounding_box(padded_hog_x1, padded_hog_y1, padded_hog_x2, padded_hog_y2, hogSize);
      title(['level : ' num2str(level) ' detlevel : ' num2str(detLevel)]);
      axis equal; axis tight; axis off;
      
      subplot(223);
      detectorIdx = bbs_nms(region_idx, 11);
      imagesc(HOGpicture(param.detectors{detectorIdx}.whow));
      axis equal; axis tight;
      
      subplot(224);
      % cla;
      % extractedHOG = hog{level}(floor(hogUIdx1):ceil(hogUIdx2), floor(hogVIdx1):ceil(hogVIdx2),:);
      % imagesc(HOGpicture(extractedHOG,hogSize));
      imagesc(HOGpicture(hog_region_pyramid{region_idx}.pyramid(pyramidIdx).padded_hog, hogSize));
      axis equal; axis tight;
      
      if level == detLevel
        fprintf('detection score single precision %f\n', bbs_nms(region_idx, 12));
        c = dwot_conv(hog_region_pyramid{region_idx}.pyramid(pyramidIdx).padded_hog,...
                                        param.detectors{detectorIdx}.whow, param);
                                      
%         c = fconvblasfloat(hog_region_pyramid{region_idx}.pyramid(pyramidIdx).padded_hog,...
%                       {single(param.detectors{detectorIdx}.whow)}, 1, 1);
        fprintf('innerproduct score double precision %f\n', max(c(:)));
      end
      waitforbuttonpress;
    end    
    pyramidIdx = pyramidIdx + 1;
  end
end

if nargout > 1 
  im_region = cell(1, n_regions);
  imSz = size(im);

  for region_idx = 1:n_regions
    clip_box = clip_to_image( round(bbs_nms(region_idx, 1:4)), [1 1 imSz(2) imSz(1)]);
    im_region{region_idx} = im(clip_box(2):clip_box(4),clip_box(1):clip_box(3),:);
  end
end


%%%%%%%% Confirmed correct 
function pad_hog = pad_hog_region(hog,...
                              hog_x1, hog_y1, hog_x2, hog_y2,...
                              padded_clip_x1, padded_clip_y1, padded_clip_x2, padded_clip_y2,...
                              xpadding, ypadding,...
                              param)
pad_hog = zeros(hog_y2 - hog_y1 + 1 + 2 * ypadding, hog_x2 - hog_x1 + 1 + 2 * xpadding, param.feature_dim, 'single');

u_start = padded_clip_y1 - hog_y1 + 1 + ypadding;
v_start = padded_clip_x1 - hog_x1 + 1 + xpadding;
if u_start < 1
  u_start = 1;
end

if v_start < 1
  v_start = 1;
end

u_size = padded_clip_y2 - padded_clip_y1;
v_size = padded_clip_x2 - padded_clip_x1;
pad_hog(u_start:(u_start + u_size), v_start:(v_start + v_size),:) = hog(padded_clip_y1:padded_clip_y2,padded_clip_x1:padded_clip_x2,:);

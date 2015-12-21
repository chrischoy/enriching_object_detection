function [curfeats, im_scale] = eod_initialize_template(I, bbox, model)
%EOD_INITIALIZE_TEMPLATE create a HOG template given an image, a bounding 
% box and a model.

sbin = model.rendering_sbin;
ncell = model.n_cell_limit;

%Expand the bbox to have some minimum and maximum aspect ratio
%constraints (if it it too horizontal, expand vertically, etc)
bbox = max(bbox,1);
bbox([1 3]) = min(size(I,2),bbox([1 3]));
bbox([2 4]) = min(size(I,1),bbox([2 4]));

bboxWidth = bbox(4) - bbox(2);
bboxHeight = bbox(3) - bbox(1);

%Create a blank image with the exemplar inside
imSize = size(I);

%Get the hog feature pyramid for the entire image
interval = 15;

%Hardcoded maximum number of levels in the pyramid
MAXLEVELS = 200;

%Get the levels per octave from the parameters
sc = 2 ^(1/interval);

scale = zeros(1,MAXLEVELS);
feat = {};

% -------------------------------------------------------------------------------
%                                                       Search for the best size.
% -------------------------------------------------------------------------------
% This is very inefficient.
for i = 1:MAXLEVELS
  scaler = 1 / sc^(i-1);

  if ceil(bboxWidth * scaler / sbin) *...
      ceil(bboxHeight * scaler / sbin) >= 1.2 * ncell
    continue;
  end

  scale(i) = scaler;
  scaled = resizeMex(I,scale(i));

  feat{i} = features_pedro(scaled,sbin);
  feat{i} = padarray(feat{i}, [1 1 0], 0); %recover lost cells

  bndX = round((size(feat{i},2)-1)*[bbox(1)-1 bbox(3)-1]/(imSize(2)-1)) + 1;
  bndY = round((size(feat{i},1)-1)*[bbox(2)-1 bbox(4)-1]/(imSize(1)-1)) + 1;

  if (bndX(2) - bndX(1) + 1) * (bndY(2) - bndY(1) + 1) <= ncell
    im_scale = scale(i);
    curfeats = feat{i}(bndY(1):bndY(2),bndX(1):bndX(2),:);
    return;
  end
end

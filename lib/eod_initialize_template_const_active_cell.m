function [curfeats, im_scale] = dwot_initialize_template_const_active_cell(I, bbox, param)

hog_cell_threshold = param.hog_cell_threshold;
sbin = param.rendering_sbin;

%Expand the bbox to have some minimum and maximum aspect ratio
%constraints (if it it too horizontal, expand vertically, etc)
bbox = max(bbox,1);
bbox([1 3]) = min(size(I,2),bbox([1 3]));
bbox([2 4]) = min(size(I,1),bbox([2 4]));

bboxWidth = bbox(4) - bbox(2);
bboxHeight = bbox(3) - bbox(1);

%Create a blank image with the exemplar inside
imSize = size(I);
Ibox = zeros(size(I,1), size(I,2));    
Ibox(bbox(2):bbox(4), bbox(1):bbox(3)) = 1;

%Get the hog feature pyramid for the entire image
interval = 15;

%Hardcoded maximum number of levels in the pyramid
MAXLEVELS = 200;

%Get the levels per octave from the parameters
sc = 2 ^(1/interval);

scale = zeros(1,MAXLEVELS);
feat = {};


for i = 1:MAXLEVELS
  scaler = 1 / sc^(i-1);
    
  if ceil(bboxWidth * scaler / sbin) * ceil(bboxHeight * scaler / sbin) >= 5 * param.n_cell_limit
    continue;
  end
  
  scale(i) = scaler;
  scaled = resizeMex(I,scale(i));
  
  feat{i} = features_pedro(scaled,sbin);
  N_NonEmptyCells = sum(sum((sum(feat{i},3) > hog_cell_threshold)));

  feat{i} = padarray(feat{i}, [1 1 0], 0);   %recover lost cells!!!

  
  
  bndX = round((size(feat{i},2)-1)*[bbox(1)-1 bbox(3)-1]/(imSize(2)-1)) + 1;
  bndY = round((size(feat{i},1)-1)*[bbox(2)-1 bbox(4)-1]/(imSize(1)-1)) + 1;

%   bndX(1) = floor(bndX(1));
%   bndX(2) = ceil(bndX(2));
%   bndY(1) = floor(bndY(1));
%   bndY(2) = ceil(bndY(2));
  
  if N_NonEmptyCells <= param.n_cell_limit
    im_scale = scale(i);
    curfeats = feat{i}(bndY(1):bndY(2),bndX(1):bndX(2),:);
    % fprintf(1,'initialized with HOG_size = [%d %d]\n',range(bndY) + 1, range(bndX) + 1);
    return;
  end
end
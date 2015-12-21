function clipped_boxes = eod_clip_bounding_box(bbox, image_size)
%clip boxes to image (just changes the max dimensions)
n_bbox = size(bbox,1);
if n_bbox == 0
    clipped_boxes = [];
    return;
end

clipped_boxes = bbox;
imbb = [1 1 image_size(2) image_size(1)];

for i = 1:2
  clipped_boxes(:,i) = max(imbb(i),bbox(:,i));
end

for i = 3:4
  clipped_boxes(:,i) = min(imbb(i),bbox(:,i));
end


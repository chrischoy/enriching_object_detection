function [result_im, clip_bnd] = eod_draw_overlay_rendering(im, box_position, rendering, depth_mask, rendering_image_weight)

assert(numel(box_position) == 4);
if nargin < 5
    rendering_image_weight = [0.7 0.3];
end

result_im = im2double(im);
box_position = round(box_position);
bboxWidth  = box_position(3) - box_position(1);
bboxHeight = box_position(4) - box_position(2);

image_size  = size(im);

clip_bnd = eod_clip_bounding_box(box_position, image_size);

rendering_size = size(rendering);
crop_region_size = round((box_position - clip_bnd) ./ [bboxWidth, bboxHeight, bboxWidth, bboxHeight] .*...
                    [rendering_size(2), rendering_size(1), rendering_size(2), rendering_size(1)]);

% since the rendering can be cropped
crop_rendering = im2double(rendering(...
          (1 - crop_region_size(2)):(end - crop_region_size(4)),...
          (1 - crop_region_size(1)):(end - crop_region_size(3)),:));

% Crop out the result image and resize it to the rendering and transfer
% the rendering. This way, we won't see any noisy white pixels after
% the image resizing. Image resizing uses interpolation which is
% inexact for our rendering.
curr_depth = depth_mask( (1 - crop_region_size(2)):(end - crop_region_size(4)),...
                       (1 - crop_region_size(1)):(end - crop_region_size(3)));
curr_depth_mask = curr_depth > 0;
curr_depth_mask = repmat(curr_depth_mask, [1, 1, 3]);

result_im_crop = result_im( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
size_cropped_rendering = size(curr_depth);
resizeResultImCrop = imresize(result_im_crop,[size_cropped_rendering(1), size_cropped_rendering(2)]);
resizeResultImCrop(curr_depth_mask) = crop_rendering(curr_depth_mask);
rendering_transferred_result_im = resizeResultImCrop;
rendering_transferred_result_im = imresize(rendering_transferred_result_im,...
                    [clip_bnd(4)-clip_bnd(2)+1, clip_bnd(3)-clip_bnd(1)+1]);
rendering_transferred_result_im(rendering_transferred_result_im(:)>1)=1;
rendering_transferred_result_im(rendering_transferred_result_im(:)<0)=0;

% Done resizing and transfering rendering.
resizeDepth = imresize(curr_depth,...
                [clip_bnd(4)-clip_bnd(2)+1, clip_bnd(3)-clip_bnd(1)+1]);

% Interpolation might introduce artifacts
resizeDepth(resizeDepth(:)>1)=1;
resizeDepth(resizeDepth(:)<0)=0;

% Conver the depth mask to logical and use it for mask
depth_mask = resizeDepth > 0;
depth_mask = repmat(depth_mask, [1, 1, 3]);

result_im_crop = result_im( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
blend_im = result_im_crop;

blend_im(depth_mask) = result_im_crop(depth_mask) * rendering_image_weight(2) + rendering_transferred_result_im(depth_mask) * rendering_image_weight(1);

result_im(clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3),:) = blend_im;

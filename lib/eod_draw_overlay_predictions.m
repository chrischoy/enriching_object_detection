function [result_im, clipped_bounding_box ] = eod_draw_overlay_predictions(im, renderings, depth_masks, prediction_bounding_boxes, rendering_image_weight, draw_padding)

if nargin < 6
    draw_padding = 0;
end

if nargin < 5;
    rendering_image_weight = [0.7, 0.3];
end

result_im = pad_image(im2double(im), draw_padding, 1);
n_predictions = size(prediction_bounding_boxes,1);
clipped_bounding_box = zeros(n_predictions, 4);

% Create overlap image
for prediction_idx = 1:n_predictions
    
    rendering  = renderings{prediction_idx};
    depth_mask = depth_masks{prediction_idx};

    box_position = round(prediction_bounding_boxes(prediction_idx, 1:4)) + draw_padding;
    
    [result_im, clip_bnd] = eod_draw_overlay_rendering(result_im, box_position, rendering, depth_mask, rendering_image_weight);

    clipped_bounding_box(prediction_idx,:) = clip_bnd;
end 

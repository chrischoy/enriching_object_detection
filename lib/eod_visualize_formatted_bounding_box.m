function [result_im, clipped_bounding_box, text_template, text_tuples] = ...
  eod_visualize_formatted_bounding_box(im, detectors, ...
  formatted_bounding_box, color_range, text_mode, ...
  rendering_image_weight, color_map, draw_padding)
%EOD_VISUALIZE_FORMATTED_BOUNDING_BOX visualize overlaid renderings.
%   the formatted bounding box must have a template index at 11th column and 
%   score on 12 th coloumn and 1:4 are bounding box x1y1x2y2. Usage
%
%   DWOT_VISUALIZE_FORMATTED_BOUNDING_BOX(im, detectors, 
%   formatted_bounding_box, color_range, text_mode), renders image with 
%   renderings overlaid
if nargin < 6
    rendering_image_weight = [0.7 0.3];
end

if nargin < 7
    n_color = numel(color_range);
    color_map = jet(n_color);
end

if nargin < 8
    draw_padding = 0;
end

% formatted bounding box structure
% 1:4 boundinb box
% 11  detector index
% 12  score
formatted_bounding_box = formatted_bounding_box(end:-1:1,:);
n_prediction = size(formatted_bounding_box,1);
prediction_bounding_boxes = formatted_bounding_box(:,1:4);
prediction_score = formatted_bounding_box(:,12);
detector_index = formatted_bounding_box(:,11);
overlap = formatted_bounding_box(:,9);

renderings = cellfun(@(x) im2double(x.rendering_image), detectors(detector_index),...
                    'UniformOutput',false);
depth_masks = cellfun(@(x) x.rendering_depth, detectors(detector_index),...
                    'UniformOutput',false);
viewpoints = cellfun(@(x) x.az, detectors(detector_index));

[text_template, text_tuples] = get_text_template_and_tuple(text_mode, prediction_score, overlap, viewpoints);

% Get rendering overlaid image
[result_im, clipped_bounding_box ] = eod_draw_overlay_predictions(im, renderings,...
    depth_masks, prediction_bounding_boxes, rendering_image_weight, draw_padding);

if nargout == 0
    % Visualize image with rendering overlaid
    imagesc(result_im); axis equal; axis tight; axis off;

    % visualize bounding box annotations
    eod_visualize_bounding_boxes(clipped_bounding_box, prediction_score, text_template, text_tuples, color_range, color_map)
end


%% Helper function
function [text_template, text_tuples] = get_text_template_and_tuple(text_mode, prediction_score, overlap, viewpoints)

switch text_mode
case 1
    text_template = 's:%0.2f';
    text_tuples = prediction_score;
case 2
    text_template = 's:%0.2f o:%0.2f';
    if 1 < size(prediction_score,2)
        prediction_score = prediction_score';
    end
    text_tuples = [prediction_score, overlap];
case 3
    text_template = 's:%0.2f o:%0.2f v:%0.0f';
    if 1 < size(prediction_score,2)
        prediction_score = prediction_score';
    end
    overlap = formatted_bounding_box(:,9);
    text_tuples = [prediction_score, overlap, viewpoints'];
end

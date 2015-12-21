function eod_visualize_proposal_tuning(im,...
    before_tuning_box, before_tuning_score, before_tuning_rendering, before_tuning_depth,...
    after_tuning_box, after_tuning_score, after_tuning_rendering, after_tuning_depth,...
    ground_truth_bounding_boxes, param)
% Given before and after parameters, visualize a tuning result. 
%    DWOT_VISUALIZE_PROPOSAL_TUNING(im, before_box, before_score, before_rendering, before_depth,
%               after_box, after_score, after_rendering, after_depth,
%               ground_truth_boxes, param)
if isempty(before_tuning_box) || isempty(after_tuning_box)
    warning('dwot_visualize_predictions_in_quadrants: no prediction to visualize');
    return;
end

if isfield(param,'color_range')
    color_range = param.color_range;
    if isfield(param,'color_map');
        color_map = param.color_map;
    else
        color_map = jet(numel(color_range));
    end  
else
    color_range = [-inf inf];
    color_map = cool(2);
end

if isfield(param,'rendering_image_weight')
    rendering_image_weight = param.rendering_image_weight;
else
    rendering_image_weight = [0.85, 0.15];
end

text_template = 's:%0.2f o:%0.2f';

[~, ~, before_tuning_iou, ~] = eod_evaluate_prediction(...
            eod_clip_bounding_box(before_tuning_box, size(im)),...
            ground_truth_bounding_boxes, param.min_overlap);

[~, ~, after_tuning_iou, ~] = eod_evaluate_prediction(...
            eod_clip_bounding_box(after_tuning_box, size(im)),...
            ground_truth_bounding_boxes, param.min_overlap);

% before_tuning_formatted_bounding_box = dwot_predictions_to_formatted_bounding_boxes(before_tuning_box, before_tuning_prediction_score, [],  before_tuning_iou );
% after_tuning_formatted_bounding_box = dwot_predictions_to_formatted_bounding_boxes(after_tuning_box, after_tuning_prediction_score, [],  after_tuning_iou );

% Original images
subplot(221);
imagesc(im); axis equal; axis tight; axis off;

% Before Tuning 
subplot(222);
[before_im] = draw_rendering_and_box(im, before_tuning_rendering, before_tuning_depth, before_tuning_box,...
      before_tuning_score, text_template, [before_tuning_score, before_tuning_iou], rendering_image_weight, ...
      color_range, color_map);


% After Tuning 
subplot(223);
[after_im] = draw_rendering_and_box(im, after_tuning_rendering, after_tuning_depth, after_tuning_box,...
      after_tuning_score, text_template, [after_tuning_score, after_tuning_iou], rendering_image_weight, ...
      color_range, color_map);

subplot(224);
imagesc([before_im after_im]); axis equal; axis tight; axis off;
title('Before (left) and After (Right) tuning');

drawnow;

% Helper function get overlaid rendering
function [result_im] = draw_rendering_and_box(im, rendering, depth, bounding_box, score,...
                    text_template, text_tuples, rendering_image_weight,  color_range, color_map)
[result_im, clipped_bounding_box] = eod_draw_overlay_rendering(im, bounding_box, rendering, depth, rendering_image_weight);

% Draw
imagesc(result_im); axis equal; axis tight; axis off;

% Draw bounding box annotation on top
eod_visualize_bounding_boxes(clipped_bounding_box, score, ...
                      text_template, text_tuples, color_range, color_map);



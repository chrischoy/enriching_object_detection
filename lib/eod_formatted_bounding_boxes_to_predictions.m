function [prediction_boxes, prediction_scores, template_indexes, overlaps, viewpoints] = eod_formatted_bounding_boxes_to_predictions(formatted_bounding_boxes) 
% return prediction information from formatted boundign boxes
%     DWOT_PREDICTIONS_TO_FORMATTED_BOUNDING_BOXES(boxes, scores, template_indexes, overlaps, viewpoints)
n_prediction = size(formatted_bounding_boxes, 1);

if n_prediction == 0
    prediction_boxes  = []; 
    prediction_scores = [];
    template_indexes  = [];
    overlaps          = []; 
    viewpoints        = [];
    return;
end

% 1:4 prediction_boxes
% 5   hog scale of the prediction, Not Used
% 6   hog pyramid level, Not Used
% 7   y_coord in hog pyramid, Not Used
% 8   x_coord in hog pyramid, Not Used
% 9   overlaps
% 10  viewpoint
% 11  detection template_indexes
% 12  score

prediction_boxes  = formatted_bounding_boxes(:,1:4);
overlaps          = formatted_bounding_boxes(:, 9);
viewpoints        = formatted_bounding_boxes(:, 10);
template_indexes  = formatted_bounding_boxes(:, 11);
prediction_scores = formatted_bounding_boxes(:, 12);

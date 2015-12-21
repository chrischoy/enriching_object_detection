function formatted_bounding_boxes = eod_predictions_to_formatted_bounding_boxes(prediction_boxes,...
                    prediction_scores, template_indexes, overlaps, viewpoints )
% return formatted bounding boxes from prediction results
%     DWOT_PREDICTIONS_TO_FORMATTED_BOUNDING_BOXES(boxes, scores, template_indexes, overlaps, viewpoints)
n_prediction = size(prediction_boxes, 1);

if n_prediction == 0
    formatted_bounding_boxes = [];
    return;
end

% Define behaviors
if nargin < 2 || isempty(prediction_scores)
    prediction_scores = -inf * ones(n_prediction, 1);
end

if nargin < 3 || isempty(template_indexes)
    template_indexes = -inf * ones(n_prediction, 1);
end

if nargin < 4 || isempty(overlaps)
    overlaps = -inf * ones(n_prediction, 1);
end

if nargin < 5 || isempty(viewpoints)
    viewpoints = -inf * ones(n_prediction, 1);
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

formatted_bounding_boxes = zeros(n_prediction,12);
formatted_bounding_boxes(:,1:4) = prediction_boxes;

formatted_bounding_boxes(:, 9) = overlaps;
formatted_bounding_boxes(:, 10) = viewpoints;
formatted_bounding_boxes(:, 11) = template_indexes;
formatted_bounding_boxes(:, 12) = prediction_scores;

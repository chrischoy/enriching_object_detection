function [true_positive, false_positive, prediction_iou, corresponding_ground_truth_idx] =...
    eod_evaluate_prediction(...
            prediction_bounding_box, ground_truth_bounding_box, min_iou,...
            excluding_ground_truth,... % logical
            prediction_azimuth,      ground_truth_azimuth,      max_azimuth_difference,...
            prediction_azimuth_rotation_direction, prediction_azimuth_offset)
% Given ground truth, compute intersection over union and set the
% predictions to be either true positive or false positive. If additional
% inputs (viewpoint predictions and ground truth viewpoints) are given,
% set the predictions to be either true positive or false positive if IOU
% is larger than the minimum overlap threshold and viewpoint prediction is
% within the viewpoint prediction threshold
% The prediction must be ordered in descending order.

% if only two inputs are given
use_viewpoint = false;
if nargin < 3
    min_iou = 0.5;
end

if nargin < 4
    excluding_ground_truth = false(1, size(ground_truth_bounding_box,1));
end

if nargin > 4
    use_viewpoint = true;
end

% if viewpoints are also given
if nargin > 4 && nargin < 7
    % default to be 45 degree bin
    max_azimuth_difference = 22.5;
end

% 1 if rotation direction is the same as ground truth rotation direction
% (clck-wise or counter-clock wise) -1 otherwise
if nargin > 4 && nargin < 8
    prediction_azimuth_rotation_direction = 1;
end

if nargin > 4 && nargin < 9
    prediction_azimuth_offset = 0;
end

if isempty(excluding_ground_truth)
    excluding_ground_truth = false(1, size(ground_truth_bounding_box,1));
else
    assert( numel(excluding_ground_truth) == size(ground_truth_bounding_box, 1));
end

n_prediction   = size(prediction_bounding_box,1);
n_ground_truth = size(ground_truth_bounding_box,1);

assert(size(prediction_bounding_box, 1) == n_prediction);

true_positive   = false(1,n_prediction);
false_positive  = false(1,n_prediction);
prediction_iou  = zeros(1,n_prediction); % intersection over union
is_detected = false(1, n_ground_truth);

% ground truth index if the prediction is true positive
corresponding_ground_truth_idx = zeros(1,n_prediction);

for prediction_idx = 1:n_prediction
    iou_max = -inf;
    curr_prediction_bounding_box = prediction_bounding_box(prediction_idx, :);
    ground_truth_idx_of_max_overlap = 0;

    if use_viewpoint
        current_prediction_viewpoint = prediction_azimuth(prediction_idx);
    end

    % search over all objects in the image
    for ground_truth_idx = 1:n_ground_truth
        curr_ground_truth_bounding_box = ground_truth_bounding_box(ground_truth_idx,:);

        box_intersection = ...
           [max(curr_prediction_bounding_box(1),curr_ground_truth_bounding_box(1));...
            max(curr_prediction_bounding_box(2),curr_ground_truth_bounding_box(2));...
            min(curr_prediction_bounding_box(3),curr_ground_truth_bounding_box(3));...
            min(curr_prediction_bounding_box(4),curr_ground_truth_bounding_box(4))];

        intersection_width  = box_intersection(3)-box_intersection(1)+1;
        intersection_height = box_intersection(4)-box_intersection(2)+1;
        if intersection_width > 0 && intersection_height > 0                
            % compute overlap as area of intersection / area of union
            ua = (curr_prediction_bounding_box(3)-curr_prediction_bounding_box(1)+1) *...
                 (curr_prediction_bounding_box(4)-curr_prediction_bounding_box(2)+1) +...
                 (curr_ground_truth_bounding_box(3)-curr_ground_truth_bounding_box(1)+1) *...
                 (curr_ground_truth_bounding_box(4)-curr_ground_truth_bounding_box(2)+1) -...
                 (intersection_width * intersection_height);
            iou = intersection_width * intersection_height/ua;

            % Compute viewpoint angle difference
            if use_viewpoint
                azimuth_modulo = mod( ...
                    prediction_azimuth_rotation_direction * current_prediction_viewpoint + ...
                    prediction_azimuth_offset, 360 );
                view_difference = min(...
                    [abs(ground_truth_azimuth(ground_truth_idx)       - azimuth_modulo),...
                    abs(ground_truth_azimuth(ground_truth_idx) + 360 - azimuth_modulo),...
                    abs(ground_truth_azimuth(ground_truth_idx) - 360 - azimuth_modulo)]);
            end

            % Find a ground truth that the prediction has the largest
            % overlap and at the same time with correct viewpoint
            if iou >= iou_max 
                % if evaluating viewpoint and viewpoint different is larger
                % than tolerance, skip
                if use_viewpoint && view_difference > max_azimuth_difference
                    continue;
                end
                iou_max = iou;
                ground_truth_idx_of_max_overlap = ground_truth_idx;
            end
        end
    end

    % If ground truth idx of max overlap is 0, no overlapping ground truth has been 
    % found. Thus this is clearly false
    if ground_truth_idx_of_max_overlap == 0
        false_positive(prediction_idx) = true;
    % For this case, if the ground truth idx is found, then we have to check whether 
    % the ground truth will be evaluated in the AP. If not, exit
    elseif ~excluding_ground_truth(ground_truth_idx_of_max_overlap) 
        % prediction has overlap more than min_iou and the corresponding ground
        % truth is not already detected nor not in the excluding set
        if iou_max >= min_iou && ~is_detected(ground_truth_idx_of_max_overlap)
            true_positive(prediction_idx) = true;
            % false_positive(prediction_idx) = false; % already false
            corresponding_ground_truth_idx(prediction_idx) = ...
                ground_truth_idx_of_max_overlap;
            is_detected(ground_truth_idx_of_max_overlap) = true;
        else
            false_positive(prediction_idx) = true;
        end
        prediction_iou(prediction_idx) = iou_max;
    end
end

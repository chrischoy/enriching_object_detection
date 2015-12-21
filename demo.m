% function demo()  % comment the functionf or debugging

% Define default class if it is not given
if ~exist('CLASS', 'var')
  CLASS = 'car';
end

% Visualize results
visualize = true;

% Get the default parameter setting
param = eod_default_param();

% Setup path
startup;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select CAD models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[model_names, model_paths] = eod_get_cad_models('data/CAD', CLASS, [], {'3ds','obj'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate NZ-WHO detectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist('renderer','var'); renderer.delete(); end

% Initialize renderer
renderer = Renderer();
if ~renderer.initialize(model_paths,...
    param.rendering_size, param.rendering_size)
  error('fail to load model');
end

[detectors] = eod_make_detectors_grid(renderer,...
  param.azimuths, param.elevations, param.yaws, param.field_of_views,...
  1:length(model_names), CLASS, param, visualize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Detect objects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

im = imread(['data/images/', 'demo1.jpeg']);
ground_truth_bounding_boxes = [55, 88, 244, 169];

[bbsAllLevel, hog, scales] = eod_detect_gpu(im, detectors, param);

% sort them according to the score and apply NMS
bbsAllLevel = eod_return_null_padded_box(bbsAllLevel, [], 12);
proposal_formatted_bounding_boxes = esvm_nms(bbsAllLevel, param.nms_threshold);
prediction_azimuth = proposal_formatted_bounding_boxes(:,10);

if visualize
  figure(1);
  eod_visualize_predictions_in_quadrants(im,...
    proposal_formatted_bounding_boxes,...
    ground_truth_bounding_boxes, detectors, param);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fine tune using Metropolis Hastings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
disp('Start MCMC sampling.');

% Choose the number of proposals to examine
n_proposal = min(param.n_max_proposals, size(proposal_formatted_bounding_boxes,1));

[hog_region_pyramids, im_regions] = eod_extract_hog(hog, scales, detectors, ...
  proposal_formatted_bounding_boxes(1:n_proposal,:), param, im);

[best_proposals] = eod_mcmc_proposal_region(renderer, hog_region_pyramids, ...
  im_regions, detectors, param, im, false);

% extract the proposal structure
[tuned_prediction_boxes, tuned_prediction_scores, tuned_prediction_azimuth,...
    tuned_prediction_elevation, tuned_prediction_yaw, tuned_prediction_fov,...
    tuned_prediction_renderings, tuned_prediction_depths] = ...
  eod_extract_proposals(best_proposals);

% Convert the prediction to the forammted bounding box for visualization
tuned_formatted_bounding_boxes = ...
  eod_predictions_to_formatted_bounding_boxes(tuned_prediction_boxes,...
    tuned_prediction_scores, [], [], tuned_prediction_azimuth);

% Visualize results
if visualize
  [proposal_boxes, proposal_scores, proposal_template_indexes] = ...
    eod_formatted_bounding_boxes_to_predictions(...
      proposal_formatted_bounding_boxes(1:n_proposal,:));

  for proposal_idx = n_proposal:-1:1
    current_proposal_detector = detectors{proposal_template_indexes(proposal_idx)};

    eod_visualize_proposal_tuning(im,...
     proposal_boxes(proposal_idx, :), proposal_scores(proposal_idx),...
     current_proposal_detector.rendering_image, current_proposal_detector.rendering_depth,...
     tuned_prediction_boxes(proposal_idx,:), tuned_prediction_scores(proposal_idx),...
     tuned_prediction_renderings{proposal_idx}, tuned_prediction_depths{proposal_idx},...
     ground_truth_bounding_boxes, param);
  end
end

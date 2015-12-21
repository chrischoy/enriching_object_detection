function [detectors]= dwot_make_detectors(renderer, azs, els, yaws, fovs, param, visualize)
% for each of the viewpoint specified, create detectors, 
% the vectors, azs, els, yaws, fovs, all have the same length
if nargin < 7
  visualize = false;
end

 
n_templates = numel(azs);

detectors = cell(1,n_templates);

for i = 1:n_templates

  detector = eod_get_detector(renderer, azs(i), els(i), yaws(i), fovs(i), [1], 'not_supported_model_class', param);

  detectors{i} = detector;

  if visualize
    figure(1); subplot(131);
    imagesc(detector.rendering_image); axis equal; axis tight;
    % subplot(132);
    % imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
    subplot(133);
    imagesc(HOGpicture(detector.whow)); axis equal; axis tight;
    disp('press any button to continue');
    waitforbuttonpress;
  end
end

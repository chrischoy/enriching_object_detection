function [detectors, detector_table]= dwot_make_detectors_table(renderer, azs, els, yaws, fovs, param, visualize)

if nargin < 7
  visualize = false;
end

%  Container class for fast query. Hash table k
detector_table = containers.Map;

i = 1;
detectors = cell(1,numel(azs) * numel(els) * numel(fovs));

for azIdx = 1:numel(azs)
  for elIdx = 1:numel(els)
    for yawIdx = 1:numel(yaws)
      for fovIdx = 1:numel(fovs)
        elGT = els(elIdx);
        azGT = azs(azIdx);
        yawGT = yaws(yawIdx);
        fovGT = fovs(fovIdx);
        
        tic
        detector = dwot_get_detector(renderer, azGT, elGT, yawGT, fovGT, [1], 'not_supported_model_class', param);
        toc;
        detectors{i} = detector;
        detector_table( dwot_detector_key(azGT, elGT, yawGT, fovGT) ) = i;

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
        i = i + 1;    
      end
    end
  end
end


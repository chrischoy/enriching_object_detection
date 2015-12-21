function [detectors, detectors_kdtree]= dwot_make_detectors_kdtree(renderer, azs, els, yaws, fovs, param, visualize)

if nargin < 7
  visualize = false;
end

 
n_templates = numel(azs) * numel(els) * numel(fovs);

detectors = cell(1,n_templates);
detectors_points = zeros(n_templates, 4);
i = 1;
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
        detectors_points(i, : ) = [azGT, elGT, yawGT, fovGT] ;
        
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

detectors_kdtree = kdtree_build( detectors_points );

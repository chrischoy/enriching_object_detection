function [detectors, param]= dwot_make_detectors_slow_gpu(mesh_path, azs, els, yaws, fovs, param, visualize)

if nargin < 7
  visualize = false;
end

%  Container class for fast query. Hash table k
param.detector_table = containers.Map;


if ~isfield(param, 'renderer')
  renderer = Renderer();
  if ~renderer.initialize([mesh_path], 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
  end
  param.renderer = renderer;
end

i = 1;
detectors = cell(1,numel(azs) * numel(els) * numel(fovs));
try
  for azIdx = 1:numel(azs)
    for elIdx = 1:numel(els)
      for yawIdx = 1:numel(yaws)
        for fovIdx = 1:numel(fovs)
          elGT = els(elIdx);
          azGT = azs(azIdx);
          yawGT = yaws(yawIdx);
          fovGT = fovs(fovIdx);

          tic
          detector = dwot_get_detector(azGT, elGT, yawGT, fovGT, [1], 'not_supported_model_class', param)
          toc;
          detectors{i} = detector;
          param.detector_table( dwot_detector_key(azGT, elGT, yawGT, fovGT) ) = i;

          if visualize
            figure(1); subplot(131);
            imagesc(im); axis equal; axis tight;
            subplot(132);
            imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
            subplot(133);
            imagesc(HOGpicture(WHOTemplate)); axis equal; axis tight;
            disp('press any button to continue');
            waitforbuttonpress;
          end
          i = i + 1;    
        end
      end
    end
  end
catch e
  disp(e.message);
end
renderer.delete();

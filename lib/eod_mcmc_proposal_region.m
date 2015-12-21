function [best_proposals]= eod_mcmc_proposal_region(renderer, hog_region_pyramid, im_region, detectors, param, visualize)

if nargin < 6
    visualize = false;
end

n_proposal_region = numel(hog_region_pyramid);

n_batch = 1;
org_cell_limit = param.n_cell_limit;
b_fixed_model = true;

best_proposals = cell(1, n_proposal_region);

uid = floor(rand * 10000);
for region_idx = 1:n_proposal_region
  % Initialize Chain
  template_idx = hog_region_pyramid{region_idx}.template_idx;
  best_state = struct('x', [detectors{template_idx}.az, detectors{template_idx}.el, detectors{template_idx}.yaw, detectors{template_idx}.fov],...
                  'models_idx', detectors{template_idx}.model_index,... % there can be multiple models and renderings 
                  'template_size', cell(1, n_batch),...
                  'rendering_image', detectors{template_idx}.rendering_image,...
                  'rendering_depth', detectors{template_idx}.rendering_depth,...
                  'image_bbox', hog_region_pyramid{region_idx}.image_bbox,...
                  'score', hog_region_pyramid{region_idx}.det_score);

  current_state = struct('x', [detectors{template_idx}.az, detectors{template_idx}.el, detectors{template_idx}.yaw, detectors{template_idx}.fov],...
                  'models_idx', detectors{template_idx}.model_index,... % for multiple model
                  'score', hog_region_pyramid{region_idx}.det_score);

  if b_fixed_model
      renderer.setModelIndex(best_state.models_idx);
  end

  % Run Chain using Metropolis-Hastings
  for mcmc_iter = 1:param.mcmc_max_iter
    update_idx = mod( mcmc_iter - 1, 4) + 1;
    for chain_idx = 1:n_batch
      proposal_x = current_state(chain_idx).x;
      % if rand(1) < 0.1
      if ~b_fixed_model
          models_idx = ceil(numel(param.model_paths) * rand(1));
      else
          models_idx = current_state(chain_idx).models_idx;
          proposal_x(update_idx) = proposal_x(update_idx) + 5 * randn(1);
      end
      
      % The sigma matrix inversion may fail due to a larger template 
      try
        [max_score, template, template_size, ...
          rendering_image, rendering_depth, image_bbox] =...
            eod_detect_using_instant_detector(renderer, ...
              hog_region_pyramid{region_idx}, proposal_x(1), ...
              proposal_x(2), proposal_x(3), proposal_x(4), models_idx, ...
              param, im_region{region_idx}, visualize);
      catch e
        disp(e.message);
        max_score = -inf;
      end
      
      if max_score > best_state(chain_idx).score
        fprintf(sprintf('region %d iter %d : %f\n',region_idx, mcmc_iter, max_score));
        best_state(chain_idx).score = max_score;
        best_state(chain_idx).template_size = template_size;
        best_state(chain_idx).x = proposal_x;
        best_state(chain_idx).image_bbox = image_bbox;
        best_state(chain_idx).rendering_image = rendering_image;
        best_state(chain_idx).rendering_depth = rendering_depth;
      end

      % Metropolis Hastings
      acc = min(1, ...
        probability_from_score((max_score - current_state(chain_idx).score) / param.n_cell_limit * 100));
      if rand(1) < acc
        fprintf('.');
        current_state(chain_idx).x = proposal_x;
        current_state(chain_idx).score = max_score;
      end
    end
  end

  best_proposals{region_idx} = best_state;
end

param.n_cell_limit = org_cell_limit;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     unnormalized probability from score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = probability_from_score(score)
p = exp(score);

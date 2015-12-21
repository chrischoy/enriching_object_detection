function [ WHOTemplate_CG, HOGTemplate, scale, residual] = eod_nzwho_cuda(im, model)
%EOD_NZWHO_CUDA given an image template and a parameter setting, make nzwho template

padding             = model.image_padding;
hog_cell_threshold  = model.hog_cell_threshold;
n_cell_limit        = model.n_cell_limit;
Mu                  = model.hog_mu;
gammaDim            = model.hog_gamma_dim;
lambda              = model.lambda;
CG_THREASHOLD       = model.cg_threshold;
CG_MAX_ITER         = model.cg_max_iter;

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0], 1);

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
% 0 NZ-WHO
% 1 Constant # active cell in NZ-WHO
% 2 Decorrelate all but center only the non-zero cells
% 3 NZ-WHO but normalize by # of active cells
% 4 HOG feature
% 5 Whiten all
% 6 Whiten all but zero our empty cells
% 7 center non zero, whiten all, zero out empty
% 8 Similar to 7 but find bias heuristically
% 9 Decomposition, Cholesky
if (model.template_initialization_mode == 0 || ...
    model.template_initialization_mode == 2 || ...
    model.template_initialization_mode == 3 || ...
    model.template_initialization_mode == 5 || ...
    model.template_initialization_mode == 6 || ...
    model.template_initialization_mode == 7)
  [HOGTemplate, scale] = eod_initialize_template(paddedIm, bbox, model);
elseif (model.template_initialization_mode == 1)
  [HOGTemplate, scale] = eod_initialize_template_const_active_cell(paddedIm, bbox, model);
elseif (model.template_initialization_mode == 4)
  error('Refactoring not finished');
else
  error('No matching initialization method');
end

% -----------------------------------------------------
%             WHO conversion using matrix decomposition
% -----------------------------------------------------
HOGTemplateSz = size(HOGTemplate);
wHeight = HOGTemplateSz(1);
wWidth = HOGTemplateSz(2);
HOGDim = HOGTemplateSz(3);

if wHeight > model.hog_gamma_cell_size(1) || ...
    wWidth > model.hog_gamma_cell_size(2)
  error(['Template dimension too large, create a larger Gamma matrix or ',...
  'decrese the number of cells per template']);
end

% Decorrelate all HOG cells
if (model.template_initialization_mode == 2 || ...
    model.template_initialization_mode == 5 || ...
    model.template_initialization_mode == 6 || ...
    model.template_initialization_mode == 7)

    nonEmptyCells = true(HOGTemplateSz(1), HOGTemplateSz(2));
    nonEmptyCells_zero_out = (sum(abs(HOGTemplate),3) > hog_cell_threshold);
elseif (model.template_initialization_mode == 0 || ...
    model.template_initialization_mode == 1 || ...
    model.template_initialization_mode == 3 )% Decorrelate only non-zero HOG cells

    nonEmptyCells = (sum(abs(HOGTemplate),3) > hog_cell_threshold);
else
    error('No matching initialization method');
end

idxNonEmptyCells = find(nonEmptyCells);
[nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
nonEmptyRows = int32(nonEmptyRows);
nonEmptyCols = int32(nonEmptyCols);

% center all cells
muSwapDim = permute(Mu,[2 3 1]);
centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);

% 2, 7 : center only non-empty cells
if (model.template_initialization_mode == 2 || ...
    model.template_initialization_mode == 7)
    centeredHOG = bsxfun(@times, centeredHOG ,single(nonEmptyCells_zero_out));
end

permHOG = permute(centeredHOG,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2
onlyNonEmptyIdx = cell2mat(arrayfun(@(x) x + (1:HOGDim)', ...
                           HOGDim * (idxNonEmptyCells - 1),'UniformOutput',false));
nonEmptyHOG = permHOG(onlyNonEmptyIdx);

[WHO_ACTIVE_CELLS] = whiten_features(model.hog_gamma_gpu, ...
                                     single(nonEmptyHOG(:)),...
                                     nonEmptyRows, nonEmptyCols,...
                                     HOGDim, lambda);

if (model.template_initialization_mode == 3)
  WHO_ACTIVE_CELLS = WHO_ACTIVE_CELLS/nnz(nonEmptyCells);
end

WHOTemplate_CG = zeros(prod(HOGTemplateSz),1,'single');
% WHOTemplate_CG(onlyNonEmptyIdx) = gather(x_min) / double(n_non_empty_cells);
WHOTemplate_CG(onlyNonEmptyIdx) = WHO_ACTIVE_CELLS;
WHOTemplate_CG =  reshape(WHOTemplate_CG,[HOGDim, wHeight, wWidth]);
WHOTemplate_CG = permute(WHOTemplate_CG,[2,3,1]);

% whiten all but zero out empty HOG region
if (model.template_initialization_mode == 6 || model.template_initialization_mode == 7)
  WHOTemplate_CG = bsxfun(@times, WHOTemplate_CG , single(nonEmptyCells_zero_out));
end

if nargout > 4
  residual = norm(b-AGPU*x);
end

% clear r b d AGPU Ad nonEmptyHOGGPU SigmaGPU nonEmptyColsGPU nonEmptyRowsGPU x x_min r_hist r_min r_norm r_start_norm beta alpha
% wait(model.gpu);

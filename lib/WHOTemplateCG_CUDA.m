function [ WHOTemplate_CG, HOGTemplate, scale, residual] = WHOTemplateCG_CUDA(im, param)
  % ( im, scrambleKernel, Mu, Gamma_GPU, gammaDim, n_cell_limit, lambda, padding, hog_cell_threshold, CG_THREASHOLD, CG_MAX_ITER, N_THREAD_H, N_THREAD_W)
%WHOTEMPLATEDECOMP Summary of this function goes here
%   Detailed explanation goes here
% Nrow = N1

% if nargin < 11
%   N_THREAD_W = 32;
% end

% if nargin < 10
%   N_THREAD_H = 32;
% end

% if nargin < 9
%   CG_MAX_ITER = 6 * 10^1;
% end

% if nargin < 8
%   CG_THREASHOLD = 10^-3;
% end

% if nargin < 7
%   hog_cell_threshold = 1.5 * 10^0;
% end

% if nargin < 6
%   padding = 50;
% end

padding             = param.image_padding;
hog_cell_threshold  = param.hog_cell_threshold;
n_cell_limit        = param.n_cell_limit;
Mu                  = param.hog_mu;
gammaDim            = param.hog_gamma_dim;
lambda              = param.lambda;
CG_THREASHOLD       = param.cg_threshold;
CG_MAX_ITER         = param.cg_max_iter;
%%%%%%%% Get HOG template

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0], 1);

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
if (param.template_initialization_mode == 0 || param.template_initialization_mode == 2 || param.template_initialization_mode == 3 )
  [HOGTemplate, scale] = eod_initialize_template(paddedIm, bbox, param);
else
  [HOGTemplate, scale] = eod_initialize_template_const_active_cell(paddedIm, bbox, param);
end

%%%%%%%% WHO conversion using matrix decomposition

HOGTemplateSz = size(HOGTemplate);
wHeight = HOGTemplateSz(1);
wWidth = HOGTemplateSz(2);
HOGDim = HOGTemplateSz(3);

if wHeight > param.hog_gamma_cell_size(1) || wWidth > param.hog_gamma_cell_size(2)
  error('Template dimension too large, create large Gamma matrix or decrese the number of cells per template');
end

% Decorrelate all HOG  cells
if (param.template_initialization_mode == 2)
    nonEmptyCells = true(HOGTemplateSz(1), HOGTemplateSz(2));
else % Decorrelate only non-zero HOG cells
    nonEmptyCells = (sum(abs(HOGTemplate),3) > hog_cell_threshold);
end

idxNonEmptyCells = find(nonEmptyCells);
[nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
nonEmptyRows = int32(nonEmptyRows);
nonEmptyCols = int32(nonEmptyCols);

muSwapDim = permute(Mu,[2 3 1]);
centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);

if (param.template_initialization_mode == 2)
    centeredHOG = centeredHOG .* repmat(double(nonEmptyCells),[1 1 HOGDim]);
end

permHOG = permute(centeredHOG,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2
onlyNonEmptyIdx = cell2mat(arrayfun(@(x) x + (1:HOGDim)', HOGDim * (idxNonEmptyCells - 1),'UniformOutput',false));
nonEmptyHOG = permHOG(onlyNonEmptyIdx);

[WHO_ACTIVE_CELLS] = whiten_features(param.hog_gamma_gpu, single(nonEmptyHOG(:)),nonEmptyRows, nonEmptyCols, HOGDim, lambda);

if (param.template_initialization_mode == 3)
  WHO_ACTIVE_CELLS = WHO_ACTIVE_CELLS/nnz(nonEmptyCells);
end

WHOTemplate_CG = zeros(prod(HOGTemplateSz),1,'single');
% WHOTemplate_CG(onlyNonEmptyIdx) = gather(x_min) / double(n_non_empty_cells);
WHOTemplate_CG(onlyNonEmptyIdx) = WHO_ACTIVE_CELLS;
WHOTemplate_CG = reshape(WHOTemplate_CG,[HOGDim, wHeight, wWidth]);
WHOTemplate_CG = permute(WHOTemplate_CG,[2,3,1]);

if nargout > 4
  residual = norm(b-AGPU*x);
end

% clear r b d AGPU Ad nonEmptyHOGGPU SigmaGPU nonEmptyColsGPU nonEmptyRowsGPU x x_min r_hist r_min r_norm r_start_norm beta alpha
% wait(param.gpu);

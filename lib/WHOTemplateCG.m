function [ WHOTemplate, HOGTemplate, scale, r_hist, residual] = WHOTemplateCG( im, param)
%WHOTEMPLATEDECOMP Summary of this function goes here
%   Detailed explanation goes here
% Nrow = N1
padding             = param.image_padding;
hog_cell_threshold  = param.hog_cell_threshold;
n_cell_limit        = param.n_cell_limit;
Mu                  = param.hog_mu;
% Gamma_GPU           = param.hog_gamma_gpu;
gammaDim            = param.hog_gamma_dim;
lambda              = param.lambda;
CG_THREASHOLD       = param.cg_threshold;
CG_MAX_ITER         = param.cg_max_iter;

%%%%%%%% Get HOG template

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0], 1);
% paddedIm(:,1:padding,:) = 1;
% paddedIm(:,end-padding+1 : end, :) = 1;
% paddedIm(1:padding,:,:) = 1;
% paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
[HOGTemplate, scale] = dwot_initialize_template(paddedIm, bbox, param);

%%%%%%%% WHO conversion using matrix decomposition

HOGTemplatesz = size(HOGTemplate);
wHeight = HOGTemplatesz(1);
wWidth = HOGTemplatesz(2);
HOGDim = HOGTemplatesz(3);
nonEmptyCells = (sum(abs(HOGTemplate),3) > hog_cell_threshold);
idxNonEmptyCells = find(nonEmptyCells);
[nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);

n_non_empty_cells = numel(nonEmptyRows);

sigmaDim = n_non_empty_cells * HOGDim;

Sigma = zeros(sigmaDim);

for cellIdx = 1:n_non_empty_cells
  rowIdx = nonEmptyRows(cellIdx); % sub2ind([wHeight, wWidth],i,j);
  colIdx = nonEmptyCols(cellIdx);
  for otherCellIdx = 1:n_non_empty_cells
    otherRowIdx = nonEmptyRows(otherCellIdx);
    otherColIdx = nonEmptyCols(otherCellIdx);
    gammaRowIdx = abs(rowIdx - otherRowIdx) + 1;
    gammaColIdx = abs(colIdx - otherColIdx) + 1;
    Sigma((cellIdx-1)*HOGDim + 1:cellIdx * HOGDim, (otherCellIdx-1)*HOGDim + 1:otherCellIdx*HOGDim) = ...
        Gamma((gammaRowIdx-1)*HOGDim + 1 : gammaRowIdx*HOGDim , (gammaColIdx - 1)*HOGDim + 1 : gammaColIdx*HOGDim);
  end
end

muSwapDim = permute(Mu,[2 3 1]);
centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);
permHOG = permute(centeredHOG,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2
onlyNonEmptyIdx = cell2mat(arrayfun(@(x) x + (1:HOGDim)', HOGDim * (idxNonEmptyCells - 1),'UniformOutput',false));
nonEmptyHOG = permHOG(onlyNonEmptyIdx);


x = zeros(sigmaDim,1,'single');
b = nonEmptyHOG;
A = Sigma + single(lambda) * eye(sigmaDim,'single');
r = b;
r_start_norm = r' * r;
d = r;

r_norm_cache = inf;

r_hist = zeros(1, MAX_ITER,'single');
i = 0;

while i < MAX_ITER
  i = i + 1;

  r_norm = (r'*r);
  r_hist(i) = r_norm/r_start_norm;
  if r_norm_cache > r_norm
    r_norm_cache = r_norm;
    x_cache = x;
  end

  if r_norm/r_start_norm < CG_THREASHOLD
    break;
  end

  Ad = A * d;
  alpha = r_norm/(d' * Ad);
  x = x + alpha * d;
  r = r - alpha * Ad;
  beta = r'*r/r_norm;
  d = r + beta * d;
end

if i == MAX_ITER
  disp('fail to get x within threshold');
end

if (param.template_initialization_mode == 3)
  x_cache = x_cache/nnz(nonEmptyCells);
end

WHOTemplate = zeros(prod(HOGTemplatesz),1);
WHOTemplate(onlyNonEmptyIdx) = x_cache(:,1);
WHOTemplate =  reshape(WHOTemplate,[HOGDim, wHeight, wWidth]);
WHOTemplate = permute(WHOTemplate,[2,3,1]);

if nargout > 3
  residual = norm(b-A*x);
end
end

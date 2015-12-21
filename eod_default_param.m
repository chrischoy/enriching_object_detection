function param = eod_default_param()
%EOD_DEFAULT_PARAM set default parameters

% Example parameter setup.
param.azimuths       = 0:15:359;  % exclude 360 degree
param.elevations     = 0:30:30;
% param.yaws           = -15:15:15;
param.yaws           = 0;
param.field_of_views = 25;
param.sbin           = 6; % size of a HOG cell in pixel
param.rendering_sbin = 6; % size of a HOG cell in pixel

param.nms_threshold = 0.5;
param.min_overlap = 0.5;
param.max_view_difference = 22.5; % degree

%How much we pad the pyramid (to let detections fall outside the image)
param.detect_pyramid_padding = 15;

% minimum image hog length that we use for convolution
param.min_hog_length = 7;

%The maximum scale to consdider in the feature pyramid
param.detect_max_scale = 1.0;

%The minimum scale to consider in the feature pyramid
param.detect_min_scale = .02;

%Initialize framing function
init_params.features = @esvm_features;
init_params.sbin = 6;
param.init_params = init_params;

%% WHO setting
% TEMPLATE_INITIALIZATION_MODE == 0
%     Creates templates that have approximately same number of cells
% and decorrelate cells with non zero HOG values
% TEMPLATE_INITIALIZATION_MODE == 1
%     Creates templates that have approxmiately same number of active cells
% Active cells are the HOG cells whose absolute values is above the
% HOG_CELL_THRESHOLD
% TEMPLATE_INITIALIZATION_MODE == 2
%     Create templates that have approximately same number of cells but
% decorrelate all cells even including zero HOG cells
% TEMPLATE_INITIALIZATION_MODE == 3
%     Create templates that have approximately same number of cells and
%     decorrelate only non-zero cells. But normalized by the number of
%     non-zero cells
% TEMPLATE_INITIALIZATION_MODE == 4
%     Create templates that have approximately same number of cells and 
%     center the HOG feature but do not decorrelate
param.template_initialization_mode = 0;
param.rendering_size      = 700;
param.image_padding       = 50;
param.lambda              = 0.15;
param.n_level_per_octave  = 20;
param.detection_threshold = 50;
param.n_cell_limit        = 250;
param.hog_cell_threshold  = 1.5;
param.feature_dim         = 31;
% Use CUDA whitening
param.use_cuda_cg         = true;

% Statistics
stats = load('data/sumGamma_N1_40_N2_40_sbin_4_nLevel_10.mat');

param.hog_mu        = stats.mu;
param.hog_gamma     = stats.Gamma;
param.hog_gamma_gpu = gpuArray(single(param.hog_gamma));
param.hog_gamma_dim = size(param.hog_gamma);
param.hog_gamma_cell_size = size(param.hog_gamma)/31;

%% GPU Setting
param.device_id = 0;

%% CG setting
param.N_THREAD_H = 32;
param.N_THREAD_W = 32;

param.scramble_gamma_to_sigma_file = 'scramble_gamma_to_sigma';
scramble_kernel = parallel.gpu.CUDAKernel(['./bin/', param.scramble_gamma_to_sigma_file '.ptx'],...
                                          ['./src/', param.scramble_gamma_to_sigma_file '.cu']);
scramble_kernel.ThreadBlockSize  = [param.N_THREAD_H , param.N_THREAD_W , 1];
param.scramble_kernel = scramble_kernel;

param.cg_threshold = 10^-3;
param.cg_max_iter  = 60;

% CPU mode = 0
% GPU mode = 1
param.computing_mode = 1;

%% Region Extraction
param.region_extraction_padding_ratio = 0.2;
param.region_extraction_levels = 0;

% MCMC Setting
param.mcmc_max_iter = 30;
param.n_max_proposals = 10;

%% Cuda Convolution Params
% THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D
param.cuda_conv_n_threads = [8, 8, 4, 32];
param.use_fft_convolution = true;

%% Binary Search params
param.binary_search_max_depth = 1;

%% Visualization
param.color_range = [-inf, param.detection_threshold:5:120, inf];
param.color_map = cool(numel(param.color_range));

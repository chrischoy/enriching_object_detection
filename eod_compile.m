function eod_compile()
%EOD_COMPILE compile c++ and cuda codes required for the package

% Add the utility path to import cuda_compile function
addpath('util');

% Set paths (For Windows, manually set the following paths)
CUDA_PATH = '';
CUDA_LIB_PATH = '';
MATLAB_PATH = '';
MATLAB_LIB_PATH = '';

% These are example paths. Please set the paths
if ismac
    CUDA_PATH = '/usr/local/cuda/';
    CUDA_LIB_PATH = '/usr/local/cuda/lib';
    MATLAB_PATH = '/Applications/MATLAB_R2014a.app/';
    MATLAB_LIB_PATH = '/Applications/MATLAB_R2014a.app/bin/maci64';
else
    CUDA_PATH = '/usr/local/cuda/';
    CUDA_LIB_PATH = '/usr/local/cuda/lib64';
    MATLAB_PATH = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
    MATLAB_LIB_PATH = '/afs/cs/package/matlab-r2013b/matlab/r2013b/bin/glnxa64';
end

% Create a directory if it does not exist
if ~exist('bin', 'dir')
  mkdir('bin');
end

% Remove all compiled binary files
delete bin/*

% -------------------------------------------------------------------------
%                                                   Compile whitening codes
% -------------------------------------------------------------------------

cuda_compile('./src', 'whiten_features', MATLAB_PATH, CUDA_PATH, './bin', false)
cuda_compile('./src', 'cudaConvolutionFFT', MATLAB_PATH, CUDA_PATH, './bin', false)

% -------------------------------------------------------------------------
%                                            Compile Gamm to Sigma ptx code
% -------------------------------------------------------------------------
% generate a ptx that generates $\Sigma$ from $\Gamma$
!nvcc -ptx src/scramble_gamma_to_sigma.cu --output-file bin/scramble_gamma_to_sigma.ptx

% -------------------------------------------------------------------------
%                                      Compile HoG feature extraction codes
% -------------------------------------------------------------------------
% resize function
mex CXX=gcc CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' ...
  src/resizeMex.cc -outdir ./bin

% Floating point version
mex CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' src/fconvblasfloat.cc -lmwblas ...
    -outdir ./bin/

% Compile Felzenszwalb's 31D features
mex -O src/features_pedro.cc -outdir ./bin


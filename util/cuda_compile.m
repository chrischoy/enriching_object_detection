function cuda_compile(src_path, func_name, matlab_root, cuda_root, out_path, debug)
%CUDA_COMPILE general cuda compiling helper for MATLAB version < 2014a
if nargin < 6
  debug = false;
end

% TODO: For matlab version < 8.0.1, Use the following setting,
% if ~verLessThan('matlab', '8.0.1')
% http://www.mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html
%   setenv('MW_NVCC_PATH',[cudaroot '/nvcc'])
%   eval(sprintf('mex -v -largeArrayDims %s.cu',func_name));
% elseif isunix && ~ismac && verLessThan('matlab', '8.0.1')

% -----------------------------------------------------------------------------
%                                               Check cuda computing capability
% -----------------------------------------------------------------------------
% TODO, CUDA Stream if high CM
gpuInfo = gpuDevice;
fprintf('Your GPU Computing Capability %d\n', str2num(gpuInfo.ComputeCapability));

% -----------------------------------------------------------------------------
%                                                   Setup environment variables
% -----------------------------------------------------------------------------

% Set debugging flag
if debug
  nvcc_debug_flag = '-g -G -O0';
  mex_debug_flag = '-g';
else
  nvcc_debug_flag = '-O3 -DNDEBUG';
  mex_debug_flag = '';
end

if ismac
  matlab_bin_path = '/bin/maci64';
else
  matlab_bin_path = '/bin/glnxa64';
end

INCLUDE_PATH = sprintf([...
    '-I./common ',...
    '-I%s/extern/include ',...
    '-I%s/toolbox/distcomp/gpu/extern/include'],...
    matlab_root, matlab_root);
NVCC_OPTS = '-arch=sm_30 -ftz=true -prec-div=false -prec-sqrt=false';
COMPILER_OPTS = '-Xcompiler -fPIC -v';

MEX_OPTS = '-largeArrayDims';
MEX_INCLUDE_PATH = sprintf('-I%s/include', cuda_root);
MEX_LIBS = '-lcublas -lcudart -lcufft -lmwgpu';
MEX_LIBRARY_PATH = ['-L', matlab_root, matlab_bin_path];

% ------------------------------------------------------------------------------
%                                                                       Compile
% ------------------------------------------------------------------------------

% Compile the object file
compile_string = sprintf([...
    '!%s/bin/nvcc ',...
    '%s ',... % Debug flag
    '%s ',... % Compiler options
    '%s ',... % NVCC_OPTS
    '%s ',... % Include paths
    '-c %s/%s.cu --output-file %s/%s.o'], ...
    cuda_root, nvcc_debug_flag, COMPILER_OPTS, NVCC_OPTS, INCLUDE_PATH,...
    src_path, func_name, out_path, func_name);

disp(compile_string);
eval(compile_string);

compile_string = sprintf(['mex ',...
    '%s ',... % Debug flag
    '%s ',... % Mex options
    '%s/%s.o  ',... % Object file
    '%s ',... % Mex library path
    '%s ',... % Mex libraries
    '-outdir %s'],... % Out path
    mex_debug_flag, MEX_OPTS, out_path, func_name, MEX_LIBRARY_PATH,...
    MEX_LIBS, out_path);

disp(compile_string);
eval(compile_string);

% % Run system command
% !nvcc -O3 -DNDEBUG -c cudaconv.cu -Xcompiler -fPIC -I/afs/cs/package/matlab-r2013b/matlab/r2013b/extern/include -I/afs/cs/package/matlab-r2013b/matlab/r2013b/toolbox/distcomp/gpu/extern/include
% % Link object
% mex cudaconv.o -L/usr/local/cuda-6.0/lib64 -L/afs/cs/package/matlab-r2013b/matlab/r2013b/bin/glnxa64 -lcudart -lcufft -lmwgpu
% -gencode arch=compute_30,code=sm_30 

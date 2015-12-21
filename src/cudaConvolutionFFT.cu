#include <cuda.h>
#include <cufft.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
// #include "common/helper_cuda.h"
#include "cudaConvFFTData.h"
#include "cudaConvFFTData.cuh"

static bool debug = false;

enum OUT_INDEX{
    CONVOLUTION_CELL_INDEX
};

enum IN_INDEX{
    DATA_INDEX,
    MAX_KERNEL_H_INDEX,
    MAX_KERNEL_W_INDEX,
    KERNLE_CELL_INDEX,
    THREAD_SIZE_INDEX, // Optional
    GPU_INDEX          // Optional
};

////////////////////////////////////////////////////////////////////////////////
// Mex Entry
////////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "cudaConvFFTData:InvalidInput";

    /* Choose a reasonably sized number of threads for the block. */
    int THREAD_PER_BLOCK_H = 16;
    int THREAD_PER_BLOCK_W = 8;
    int THREAD_PER_BLOCK_D = 8;
    int THREAD_PER_BLOCK_2D = 32;

    /* Initialize the MathWorks GPU API. */
    // If initialized mxInitGPU do nothing
    if (mxInitGPU() != MX_GPU_SUCCESS)
        mexErrMsgTxt("mxInitGPU fail");
    

    /* Throw an error if the number of inputs mismatch */
    if ( (nrhs <  (KERNLE_CELL_INDEX + 1)) || (nrhs > (GPU_INDEX + 1) ))
        mexErrMsgIdAndTxt(errId, "Wrong number of inputs");


    /*  Set data */
    const mxArray *mxDATA = prhs[DATA_INDEX];
    if (mxIsGPUArray(mxDATA) || 
            mxGetNumberOfDimensions(mxDATA) != 3 || 
            mxGetClassID(mxDATA) != mxSINGLE_CLASS)
        mexErrMsgTxt("Invalid data input");


    /* Kernel dimensions */
    int MAX_KERNEL_H = (int)mxGetScalar(prhs[MAX_KERNEL_H_INDEX]);
    int MAX_KERNEL_W = (int)mxGetScalar(prhs[MAX_KERNEL_W_INDEX]);
    if(debug) fprintf(stderr,"Kernel size: h=%d, w=%d\n",MAX_KERNEL_H,MAX_KERNEL_W);


    /* Kernel Input */
    if (mxGetClassID(prhs[KERNLE_CELL_INDEX]) != mxCELL_CLASS)
        mexErrMsgIdAndTxt(errId, "Kernel must be a cell array");
    mwSize nKernel = mxGetNumberOfElements(prhs[KERNLE_CELL_INDEX]);
    int N_KERNEL = (int)nKernel;
    if(debug) fprintf(stderr,"N Kernel: %d\n", N_KERNEL);


    /* Thread size */
    if (( nrhs > THREAD_SIZE_INDEX)  && mxGetNumberOfElements(prhs[THREAD_SIZE_INDEX]) != 4)
        mexErrMsgIdAndTxt(errId, "CUDA Thread Size must be 4 integers : THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D\nYou must choose size such that total thread will not be larger than MaxThreadsPerBlock");

    if ( nrhs > THREAD_SIZE_INDEX ){
        const double* threadSize = (double *)mxGetData(prhs[THREAD_SIZE_INDEX]);
        THREAD_PER_BLOCK_H = (int)threadSize[0];
        THREAD_PER_BLOCK_W = (int)threadSize[1];
        THREAD_PER_BLOCK_D = (int)threadSize[2];
        THREAD_PER_BLOCK_2D = (int)threadSize[3];
        if(debug) fprintf(stderr,"Thread size: H=%d, W=%d, D=%d, 2D=%d\n", THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D);
    }

    int GPU_ID = 0;
    if (nrhs > GPU_INDEX ){
       GPU_ID = (int)mxGetScalar(prhs[GPU_INDEX]); 
       if(debug) fprintf(stderr,"Using GPU : %d\n", GPU_ID);
       cudaSetDevice(GPU_ID);
    }


    /*  FFT Data */
    // Data dimensions
    const mwSize *DATA_dims = mxGetDimensions(mxDATA);
    int DATA_H = DATA_dims[0];
    int DATA_W = DATA_dims[1];
    int FEATURE_DIM = DATA_dims[2];

    float *h_Data = (float *)mxGetData(mxDATA);
    if(debug) fprintf(stderr,"Data size: h=%d, w=%d, f=%d\n",DATA_H,DATA_W,FEATURE_DIM); 

    // Width and height of padding
    int PADDING_H = MAX_KERNEL_H - 1;
    int PADDING_W = MAX_KERNEL_W - 1;

    // Derive FFT size from data and kernel dimensions
    // FFT_H = computeFFTsize(DATA_H + PADDING_H);
    // FFT_W = computeFFTsize(DATA_W + PADDING_W);
    int FFT_H = computeFFTsize16(DATA_H + PADDING_H);
    int FFT_W = computeFFTsize16(DATA_W + PADDING_W);
    int CFFT_W = FFT_W;
    int CFFT_H = FFT_H/2 + 1;

    if(debug) fprintf(stderr,"FFT size: h=%d, w=%d\n",FFT_H,FFT_W);

    int DATA_SIZE = DATA_W * DATA_H * FEATURE_DIM * sizeof(float);
    int FFT_SIZE  = FFT_W  * FFT_H  * FEATURE_DIM * sizeof(float);
    int CFFT_SIZE = CFFT_W * CFFT_H * FEATURE_DIM * sizeof(float2);
    int CONV_SIZE = FFT_W  * FFT_H  * sizeof(float);
    
    int BATCH = FEATURE_DIM;
    int FFT_Dims[] = { FFT_W, FFT_H };
    int CFFT_Dims[] = { CFFT_W, CFFT_H };
    int idist = FFT_W * FFT_H;
    int odist = CFFT_W * CFFT_H;

    cufftHandle FFTplan_R2C, FFTplan_C2R;
    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_R2C, 
        2, // rank
        FFT_Dims, 
        FFT_Dims, 1, idist, // *inembed, istride, idist
        CFFT_Dims, 1, odist, // *onembed, ostride, odist
        CUFFT_R2C, 
        BATCH)); // batch

    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_C2R, 
        2, // rank
        FFT_Dims,
        CFFT_Dims, 1, odist, // *inembed, istride, idist
        FFT_Dims, 1, idist, // *onembed, ostride, odist
        CUFFT_C2R, 
        BATCH)); // batch

    float *d_Data;
    float *d_PaddedData;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_Data,         DATA_SIZE));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_PaddedData,   FFT_SIZE));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice));

    dim3 threadBlock3D(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    dim3 dataBlockGrid3D( iDivUp(FFT_W, threadBlock3D.x), 
                        iDivUp(FFT_H, threadBlock3D.y), 
                        iDivUp(FEATURE_DIM, threadBlock3D.z));

    padData<<<dataBlockGrid3D, threadBlock3D>>>(
        d_PaddedData,
        d_Data,
        FFT_W,
        FFT_H,
        DATA_W,
        DATA_H,
        FEATURE_DIM
        );

    cufftComplex *d_CFFT_DATA;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_CFFT_DATA,     CFFT_SIZE));
    CUFFT_SAFE_CALL(cufftExecR2C(FFTplan_R2C, d_PaddedData, d_CFFT_DATA));
    CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
    cudaFree(d_Data);




    

    /* Convolution FFT */
    // Set Variables 
    float *d_IFFTEProd;
    float *d_CONVOLUTION;
    cufftComplex *d_CFFT_KERNEL;
    cufftComplex *d_FFTEProd;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_IFFTEProd,    FFT_SIZE));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_CONVOLUTION,  CONV_SIZE));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_CFFT_KERNEL,  CFFT_SIZE));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_FFTEProd,     CFFT_SIZE));
    
    const mxArray *mxCurrentCell;
    const mxGPUArray *mxKernel;
    const mwSize *mxKernel_Dim;
    float *h_Kernel;
    float *d_Kernel;
    int KERNEL_H, KERNEL_W, KERNEL_SIZE;

    dim3 threadBlock2D( THREAD_PER_BLOCK_2D, THREAD_PER_BLOCK_2D);
    dim3 dataBlockGrid2D( iDivUp(FFT_W, threadBlock2D.x), 
                        iDivUp(FFT_H, threadBlock2D.y));
    
    mwSize mwCONV_Dims[2];
    mwCONV_Dims[0] = FFT_H;
    mwCONV_Dims[1] = FFT_W;

    plhs[CONVOLUTION_CELL_INDEX] = mxCreateCellMatrix(1, N_KERNEL);

    for (int kernelIdx = 0; kernelIdx < N_KERNEL; kernelIdx++){
        
        // Get Kernel Data
        mxCurrentCell = mxGetCell(prhs[KERNLE_CELL_INDEX], kernelIdx);
        if (!mxIsGPUArray(mxCurrentCell)){
            
            if( mxGetClassID(mxCurrentCell) != mxSINGLE_CLASS || mxGetNumberOfDimensions(mxCurrentCell) != 3 )
                mexErrMsgIdAndTxt(errId, "Kernels must be of type float and have features larger than 1");

            h_Kernel = (float *)mxGetData(mxCurrentCell);
            mxKernel_Dim = mxGetDimensions(mxCurrentCell);

            // Kernel dimensions
            KERNEL_H = mxKernel_Dim[0];
            KERNEL_W = mxKernel_Dim[1];
            KERNEL_SIZE = KERNEL_W * KERNEL_H * FEATURE_DIM * sizeof(float);

            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_Kernel, KERNEL_SIZE));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
            mxKernel = NULL;
        }else{ // Kernel is GPU Array
            mxKernel = mxGPUCreateFromMxArray(mxCurrentCell);

            if ( mxGPUGetClassID(mxKernel) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(mxKernel) != 3 )
                mexErrMsgIdAndTxt(errId, "Kernels must be of type float and have features larger than 1");

            mxKernel_Dim = mxGPUGetDimensions(mxKernel);

            // Kernel dimensions
            KERNEL_H = mxKernel_Dim[0];
            KERNEL_W = mxKernel_Dim[1];
            KERNEL_SIZE = KERNEL_W * KERNEL_H * FEATURE_DIM * sizeof(float);

            d_Kernel = (float *)mxGPUGetDataReadOnly(mxKernel);
        }

        if(debug) fprintf(stderr,"Kernel size: h=%d, w=%d\n", KERNEL_H, KERNEL_W);

        if (FEATURE_DIM != mxKernel_Dim[2] || KERNEL_W > FFT_W || KERNEL_H > FFT_H )
            mexErrMsgIdAndTxt(errId, "Kernel and Data must have the same number of features and kernel size should be smaller than data size");

        padData<<<dataBlockGrid3D, threadBlock3D>>>(
                d_PaddedData,
                d_Kernel,
                FFT_W,
                FFT_H,
                KERNEL_W,
                KERNEL_H,
                FEATURE_DIM
            );

        CUFFT_SAFE_CALL(cufftExecR2C(FFTplan_R2C, d_PaddedData, d_CFFT_KERNEL));
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        if(debug) fprintf(stderr,"FFT done\n");

        
        /* Hadamard product, Element-wise multiplication in frequency domain */
        /* If execute the following, second compile of this file create MATLAB error */
        elementwiseProductAndNormalize<<<dataBlockGrid3D, threadBlock3D>>>(
                d_FFTEProd, // out
                d_CFFT_DATA, // in data
                d_CFFT_KERNEL,   // in kernel
                CFFT_H,
                CFFT_W,
                FEATURE_DIM,
                1.0f / (FFT_W * FFT_H)
            );

        CUFFT_SAFE_CALL(cufftExecC2R(FFTplan_C2R, d_FFTEProd, d_IFFTEProd));
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        sumAlongFeatures<<<dataBlockGrid2D, threadBlock2D>>>(
                d_CONVOLUTION,
                d_IFFTEProd,
                FFT_H,
                FFT_W,
                FEATURE_DIM
            );

        mxArray * convolutionResult = mxCreateNumericArray(2, mwCONV_Dims, mxSINGLE_CLASS, mxREAL);
        float * h_CONVOLUTION = (float *)mxGetData(convolutionResult);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_CONVOLUTION, d_CONVOLUTION, CONV_SIZE ,cudaMemcpyDeviceToHost));

        mxSetCell(plhs[CONVOLUTION_CELL_INDEX], kernelIdx, convolutionResult);
        if(mxKernel == NULL) cudaFree(d_Kernel);
        else mxGPUDestroyGPUArray(mxKernel);
    }
    // plhs[1] = mxGPUCreateMxArrayOnGPU(mxFFTKernel);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    // mxGPUDestroyGPUArray(mxFFTData);
    // mxGPUDestroyGPUArray(mxConvolution);    
    // mxGPUDestroyGPUArray(mxFFTKernel);
    
    cufftDestroy(FFTplan_R2C);
    cufftDestroy(FFTplan_C2R);

    cudaFree(d_CFFT_DATA);
    cudaFree(d_IFFTEProd);
    cudaFree(d_CONVOLUTION);
    cudaFree(d_CFFT_KERNEL);
    cudaFree(d_FFTEProd);
    cudaFree(d_PaddedData);
}

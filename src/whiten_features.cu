#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Matlab libraries
#include "mex.h"
#include "gpu/mxGPUArray.h"

// local dependencies
#include "whiten_features.h"
#include "whiten_features.cuh"

static bool debug = false;

/*
    int GAMMA_IN            = 0;
    int CENTERED_TEMPLATE_IN= 1;
    int NON_EMPTY_ROW_IN    = 2;
    int NON_EMPTY_COL_IN    = 3;
    int FEATURE_DIM_IN      = 4;
    int LAMBDA_IN           = 5;
    int FEATURE_THRESHOLD_IN= 6;   // Optional 
    int CG_TOLERANCE_IN     = 7;   // Optional 
    int CG_MAX_ITER_IN      = 8;   // Optional 
    int THREAD_SIZE_IN      = 9;   // Optional 
*/
////////////////////////////////////////////////////////////////////////////////
// Mex Entry
////////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "cudaDecorrelateFeature:InvalidInput";

    float FEATURE_THRESHOLD = 1.5f;
    float CG_TOLERANCE      = 0.01f;
    int CG_MAX_ITER         = 60;

    /* Choose a reasonably sized number of threads for the block. */
    int THREAD_PER_BLOCK_H  = 16;
    int THREAD_PER_BLOCK_W  = 8;
    int THREAD_PER_BLOCK_D  = 8;
    int THREAD_PER_BLOCK_2D_H = 32;
    int THREAD_PER_BLOCK_2D_W = 32;

    /* Initialize the MathWorks GPU API. */
    // If initialized mxInitGPU do nothing
    if (mxInitGPU() != MX_GPU_SUCCESS)
        mexErrMsgTxt("mxInitGPU fail");

    int CG_OUT      = 0;
    int SIGMA_OUT   = 1;

    int GAMMA_IN             = 0;
    int CENTERED_TEMPLATE_IN = 1;
    int NON_EMPTY_ROW_IN     = 2;
    int NON_EMPTY_COL_IN     = 3;
    int FEATURE_DIM_IN       = 4;
    int LAMBDA_IN            = 5;
    int FEATURE_THRESHOLD_IN = 6;   // Optional 
    int CG_TOLERANCE_IN      = 7;   // Optional 
    int CG_MAX_ITER_IN       = 8;   // Optional 
    int THREAD_SIZE_IN       = 9;   // Optional 

    if ( (nrhs < FEATURE_THRESHOLD_IN ) || (nrhs > (THREAD_SIZE_IN + 1) ) )
        mexErrMsgIdAndTxt(errId, "Wrong number of inputs");
    

    /* Gamma */
    if ( !mxIsGPUArray(prhs[GAMMA_IN]))
        mexErrMsgIdAndTxt(errId, "The Gamma must be real single precision array in GPU");
    const mxGPUArray *mxGamma = mxGPUCreateFromMxArray(prhs[GAMMA_IN]);
    if ( mxGPUGetClassID(mxGamma) != mxSINGLE_CLASS )
        mexErrMsgIdAndTxt(errId, "The Gamma must be real single precision array in GPU");
    const mwSize *mxGamma_Dim = mxGPUGetDimensions(mxGamma);
    const int GammaDim        = mxGamma_Dim[0];
    const float *d_Gamma      = (float *)mxGPUGetDataReadOnly(mxGamma);


    /* Centered Template */
    const mxArray *mxCenteredTemplate = prhs[CENTERED_TEMPLATE_IN];
    if ( mxGetClassID(mxCenteredTemplate) != mxSINGLE_CLASS )
        mexErrMsgTxt("Invalid input: hog template");
    float * h_centered_template = (float *)mxGetPr(mxCenteredTemplate);
    
    /* Template height */
    // if (mxGetClassID(prhs[TEMPLATE_HEIGHT_IN]) != mxDOUBLE_CLASS)
    //     mexErrMsgTxt("Invalid input: template height");
    // TEMPLATE_HEIGHT = (float)mxGetScalar(prhs[TEMPLATE_HEIGHT_IN]);


    /* Template height */
    // if (mxGetClassID(prhs[TEMPLATE_WIDTH_IN]) != mxDOUBLE_CLASS)
    //     mexErrMsgTxt("Invalid input: template width");
    // TEMPLATE_WIDTH = (float)mxGetScalar(prhs[TEMPLATE_WIDTH_IN]);


    /* Feature dimension */
    if (mxGetClassID(prhs[FEATURE_DIM_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: feature dimension");
    int FEATURE_DIM = (float)mxGetScalar(prhs[FEATURE_DIM_IN]);


    /* Non Empty Col and Row Index */
    const mxArray *mxNonEmptyRows       = prhs[NON_EMPTY_ROW_IN];
    const mxArray *mxNonEmptyCols       = prhs[NON_EMPTY_COL_IN];
    const mwSize  *mxNonEmptyRowsDim    = mxGetDimensions(mxNonEmptyRows);
    const mwSize  *mxNonEmptyColsDim    = mxGetDimensions(mxNonEmptyCols);
    if( mxNonEmptyRowsDim[0] != mxNonEmptyColsDim[0] ||
        mxNonEmptyRowsDim[1] != mxNonEmptyColsDim[1] ||
        mxGetClassID(mxNonEmptyRows) != mxINT32_CLASS ||
        mxGetClassID(mxNonEmptyCols) != mxINT32_CLASS)
        mexErrMsgIdAndTxt(errId, "Invalid non empty indexes");
    int N_ACTIVE_CELL = max(mxNonEmptyRowsDim[0], mxNonEmptyRowsDim[1]);
    int *h_nonEmptyRows = (int*)mxGetPr(mxNonEmptyRows);
    int *h_nonEmptyCols = (int*)mxGetPr(mxNonEmptyCols);


    /* Lambda, added to the diagonals of Sigma */
    if (mxGetClassID(prhs[LAMBDA_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: lambda");
    float lambda = (float)mxGetScalar(prhs[LAMBDA_IN]);


    /* FEATURE_THRESHOLD */
    if (nrhs > FEATURE_THRESHOLD_IN && mxGetClassID(prhs[FEATURE_THRESHOLD_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: feature threshold");
    if (nrhs > FEATURE_THRESHOLD_IN)
        FEATURE_THRESHOLD = (float)mxGetScalar(prhs[FEATURE_THRESHOLD_IN]);


    /* CG_TOLERANCE */
    if (nrhs > CG_TOLERANCE_IN && mxGetClassID(prhs[CG_TOLERANCE_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: CG_TOLERANCE");
    if (nrhs > CG_TOLERANCE_IN)
        CG_TOLERANCE = (float)mxGetScalar(prhs[CG_TOLERANCE_IN]);


    /* CG_MAX_ITER */
    if (nrhs > CG_MAX_ITER_IN && mxGetClassID(prhs[CG_MAX_ITER_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: CG_MAX_ITER");
    if (nrhs > CG_MAX_ITER_IN)
        CG_MAX_ITER = (int)mxGetScalar(prhs[CG_MAX_ITER_IN]);



    /* Check the Thread Size Parameters */
    // THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D_H, THREAD_PER_BLOCK_2D_W
    if ( nrhs > THREAD_SIZE_IN  && mxGetNumberOfElements(prhs[THREAD_SIZE_IN]) != 5)
        mexErrMsgIdAndTxt(errId, "CUDA Thread size invalid format");

    if ( nrhs > THREAD_SIZE_IN ){
        const double* threadSize = (double *)mxGetData(prhs[THREAD_SIZE_IN]);
        THREAD_PER_BLOCK_H = (int)threadSize[0];
        THREAD_PER_BLOCK_W = (int)threadSize[1];
        THREAD_PER_BLOCK_D = (int)threadSize[2];
        THREAD_PER_BLOCK_2D_H = (int)threadSize[3];
        THREAD_PER_BLOCK_2D_W = (int)threadSize[4];
        if(debug) fprintf(stderr,"Thread size: H=%d, W=%d, D=%d, 2D_H=%d, 2D_W=%d\n",
                                    THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, 
                                    THREAD_PER_BLOCK_2D_H, THREAD_PER_BLOCK_2D_W);
    }


    // cudaDeviceProp dev;
    // cudaGetDeviceProperties(&dev,0);
    // int success = checkDeviceProp(dev);
    
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);
    /* Find number of cuda capable devices */
    // CUDA_SAFE_CALL(cudaGetDeviceCount(&N_GPU));
    // if(debug) fprintf(stderr, "CUDA-capable device count: %i\n", N_GPU);

    /* Setup Variables */
    int N = N_ACTIVE_CELL * FEATURE_DIM;

    /* Set block size and thread size */
    // dim3 threadBlock3D(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    // dim3 dataBlockGrid3D( iDivUp(WIDTH, threadBlock3D.x), 
    //                       iDivUp(HEIGHT, threadBlock3D.y), 
    //                       iDivUp(FEATURE_DIM, threadBlock3D.z));

    dim3 threadBlock2D( THREAD_PER_BLOCK_2D_W, THREAD_PER_BLOCK_2D_H);
    dim3 dataBlockGrid2D( iDivUp(N, threadBlock2D.x), 
                          iDivUp(N, threadBlock2D.y));

    float * d_Sigma;
    cudaMalloc((void **)&d_Sigma, N * N * sizeof(float));

    // thrust::device_vector<float> vec_d_Sigma(N * N);
    // float* d_Sigma  = thrust::raw_pointer_cast(&vec_d_Sigma[0]);

    int * d_nonEmptyRows;
    int * d_nonEmptyCols;
    cudaMalloc((void **)&d_nonEmptyRows, N_ACTIVE_CELL * sizeof(float));
    cudaMalloc((void **)&d_nonEmptyCols, N_ACTIVE_CELL * sizeof(float));
    cudaMemcpy(d_nonEmptyRows, h_nonEmptyRows, N_ACTIVE_CELL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonEmptyCols, h_nonEmptyCols, N_ACTIVE_CELL * sizeof(float), cudaMemcpyHostToDevice);

    // thrust::device_vector<int> vec_d_nonEmptyRows(h_nonEmptyRows, h_nonEmptyRows + N_ACTIVE_CELL);
    // thrust::device_vector<int> vec_d_nonEmptyCols(h_nonEmptyCols, h_nonEmptyCols + N_ACTIVE_CELL);
    // int* d_nonEmptyRows  = thrust::raw_pointer_cast(&vec_d_nonEmptyRows[0]);
    // int* d_nonEmptyCols  = thrust::raw_pointer_cast(&vec_d_nonEmptyCols[0]);
    scrambleGammaToSigma<<<dataBlockGrid2D, threadBlock2D>>>( d_Sigma,
                                d_Gamma,
                                lambda,
                                d_nonEmptyRows, 
                                d_nonEmptyCols,
                                GammaDim, FEATURE_DIM, N_ACTIVE_CELL );
    

    /////////////////////////////////////////////
    ///    DEBUG
    /////////////////////////////////////////////
    if (debug){
        mwSize mwSigma[2];
        mwSigma[0] = N; mwSigma[1] = N;
        plhs[SIGMA_OUT] = mxCreateNumericArray(2, mwSigma, mxSINGLE_CLASS, mxREAL);
        float* h_Sigma = (float *)mxGetData(plhs[SIGMA_OUT]);
        cudaMemcpy(h_Sigma, d_Sigma, N * N * sizeof(float) ,cudaMemcpyDeviceToHost);
    }
    /////////////////////////////////////////////


    float * d_x;
    float * d_r;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_r, N * sizeof(float));
    cudaMemset(d_x, 0, N * sizeof(float));
    cudaMemcpy(d_r, h_centered_template, N * sizeof(float), cudaMemcpyHostToDevice);

    decorrelateFeature(cublasHandle, d_x, d_Sigma, d_r, N, CG_TOLERANCE, CG_MAX_ITER);

    mwSize mwN[1];
    mwN[0] = N;
    plhs[CG_OUT] = mxCreateNumericArray(1, mwN, mxSINGLE_CLASS, mxREAL);
    float* h_CG = (float *)mxGetData(plhs[CG_OUT]);
    cudaMemcpy(h_CG, d_x, N * sizeof(float) ,cudaMemcpyDeviceToHost);

    cudaFree(d_Sigma);
    cudaFree(d_nonEmptyRows);
    cudaFree(d_nonEmptyCols);
    cudaFree(d_x);
    cudaFree(d_r);

    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(mxGamma);
    // mxFree(mwN);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    // cudaDeviceReset();
}

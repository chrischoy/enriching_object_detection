#ifndef CUDA_CONV_FFT_DATA_CUH
#define CUDA_CONV_FFT_DATA_CUH

/*
 * Device Code
 */

////////////////////////////////////////////////////////////////////////////////
// Pad data with zeros, 
////////////////////////////////////////////////////////////////////////////////
__global__ void padData(
    float *d_PaddedData,
    const float *d_Data,
    int fftW,
    int fftH,
    int dataW,
    int dataH,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    if(x < fftW && y < fftH && z < FEATURE_DIM){
        if(x < dataW && y < dataH)
            d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
                    d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
        else
            d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
__device__ void complexMulAndScale(cufftComplex &out, cufftComplex a, cufftComplex b, float c){
    const cufftComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    out = t;
}

__device__ void complexConjMulAndScale(cufftComplex &out, cufftComplex a, cufftComplex b, float c){
    const cufftComplex t = {c * (a.x * b.x + a.y * b.y), c * (a.y * b.x - a.x * b.y)};
    out = t;
}

__global__ void elementwiseProductAndNormalize(
    cufftComplex *fft_Output,
    const cufftComplex *fft_PaddedData,
    const cufftComplex *fft_PaddedKernel,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM,
    float scale
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;
    
    if(x < FFT_W && y < FFT_H && z < FEATURE_DIM){
        // int i = IMUL(z, IMUL(FFT_W, FFT_H)) + IMUL(FFT_H, x) + y;
        int i = z * FFT_W * FFT_H + FFT_H * x + y;
        // complexConjMulAndScale(fft_Output[i], fft_PaddedData[i], fft_PaddedKernel[i], scale);
        fft_Output[i].x = scale * (fft_PaddedData[i].x * fft_PaddedKernel[i].x - fft_PaddedData[i].y * fft_PaddedKernel[i].y);
        fft_Output[i].y = scale * (fft_PaddedData[i].y * fft_PaddedKernel[i].x + fft_PaddedData[i].x * fft_PaddedKernel[i].y);
    }
}

/* Support in-place computation, i.e. input and output can be the same */
__global__ void sumAlongFeatures(
    float *convolutionResult,
    const float *convolutionPerFeature,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;

    if(x < FFT_W && y < FFT_H){
        const int result_i = IMUL(FFT_H, x) + y;
        const int N = IMUL(FFT_W, FFT_H);

        float acc = convolutionPerFeature[result_i];
        int zN = N;
        for (int z = 1; z < FEATURE_DIM; z++){
            acc += convolutionPerFeature[zN + result_i];
            zN += N;
        }
        convolutionResult[result_i] = acc;
    }
}
    

#endif
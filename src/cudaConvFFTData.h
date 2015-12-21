#ifndef CUDA_CONV_FFT_DATA
#define CUDA_CONV_FFT_DATA

#  define IMUL(a, b) __mul24(a, b)

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        printf("Cuda error in file '%s' in line %i Error : %d.\n",            \
                __FILE__, __LINE__, err);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        printf("Cuda error in file '%s' in line %i Error : %d.\n",            \
                __FILE__, __LINE__,err);                                        \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUFFT_SAFE_CALL( call) do {                                        \
    cufftResult err = call;                                                  \
    if( CUFFT_SUCCESS != err) {                                              \
        printf("CUFFT error in file '%s' in line %i Error : %d.\n",            \
                __FILE__, __LINE__,err);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}



int checkDeviceProp ( cudaDeviceProp p ) {
    int support = p.canMapHostMemory;

    if(support == 0) printf( "%s does not support mapping host memory.\n", p.name);
    else             printf( "%s supports mapping host memory.\n",p.name);

    support = p.concurrentKernels;
    if(support == 0) printf("%s does not support concurrent kernels\n", p.name);
    else printf("%s supports concurrent kernels\n",p.name);

    support = p.kernelExecTimeoutEnabled;
    if(support == 0) printf("%s kernelExecTimeout disabled\n", p.name);
    else printf("%s kernelExecTimeout enabled\n",p.name);

    printf("compute capability : %d.%d \n", p.major,p.minor);
    printf("number of multiprocessors : %d \n", p.multiProcessorCount);

    return support;
}

int computeFFTsize(int dataSize){
    //Highest non-zero bit position of dataSize
    int hiBit;
    //Neares lower and higher powers of two numbers for dataSize
    unsigned int lowPOT, hiPOT;

    //Align data size to a multiple of half-warp
    //in order to have each line starting at properly aligned addresses
    //for coalesced global memory writes in padKernel() and padData()
    dataSize = iAlignUp(dataSize, 16);

    //Find highest non-zero bit
    for(hiBit = 31; hiBit >= 0; hiBit--)
        if(dataSize & (1U << hiBit)) break;

    //No need to align, if already power of two
    lowPOT = 1U << hiBit;
    if(lowPOT == dataSize) return dataSize;

    //Align to a nearest higher power of two, if the size is small enough,
    //else align only to a nearest higher multiple of 512,
    //in order to save computation and memory bandwidth
    hiPOT = 1U << (hiBit + 1);
    //if(hiPOT <= 1024)
        return hiPOT;
    //else 
    //  return iAlignUp(dataSize, 512);
}

int computeFFTsize16(int dataSize){
    // Compute the multiple of 16
    int mod = dataSize / 16;
    int rem = dataSize % 16;

    return (mod * 16) + ((rem > 0)?16:0);
}

#endif
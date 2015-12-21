#ifndef CUDA_DECORRELATE_FEATURE_H
#define CUDA_DECORRELATE_FEATURE_H

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
   
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/* 
    cublasHandle : cublasHandle, input
    d_x : initialized pointer,   ouput
    d_Sigma : Covariance matrix, input
    d_r : Ax = b, r = b pointer, input
    N : Dimension of the vector, input
    CG_TOLERANCE : input
    CG_MAX_ITER  : input
*/
void decorrelateFeature(cublasHandle_t cublasHandle, float* d_x, float* d_Sigma, float* d_r, int N, float CG_TOLERANCE, int CG_MAX_ITER){
    float * d_p;
    float * d_Ax;
    cudaMalloc((void **)&d_p, N * sizeof(float));
    cudaMalloc((void **)&d_Ax, N * sizeof(float));

    // // Initial point is at the origin
    // thrust::device_vector<float> vec_d_x(N, 0);
    // thrust::device_vector<float> vec_d_r(h_centered_template, h_centered_template + N);
    // thrust::device_vector<float> vec_d_p(N);
    // thrust::device_vector<float> vec_d_Ax(N);

    // float* d_x  = thrust::raw_pointer_cast(&vec_d_x[0]);
    // float* d_r  = thrust::raw_pointer_cast(&vec_d_r[0]);
    // float* d_p  = thrust::raw_pointer_cast(&vec_d_p[0]);
    // float* d_Ax = thrust::raw_pointer_cast(&vec_d_Ax[0]);

    float alpha     = 1.0f;
    float alpham1   = -1.0f;
    float beta      = 0.0f;
    float a, b, na, r0, r1, dot;

    cublasStatus_t cublasStatus;

    // Ax = A * x
    // y = α op(A) * x + β * y
    cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, 
                N, N, 
                &alpha,
                d_Sigma, N,
                d_x, 1,
                &beta, 
                d_Ax, 1);
    // checkCudaErrors(cublasStatus);

    // r = -A * x = - Ax
    // y = α x + y
    cublasStatus = cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    // checkCudaErrors(cublasStatus);

    // r1 = r^T r
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    // checkCudaErrors(cublasStatus);

    int k = 1;

    while (r1 > CG_TOLERANCE && k <= CG_MAX_ITER)
    {
        if (k > 1)
        {
            b = r1 / r0;
            // p = bp
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            // checkCudaErrors(cublasStatus);
            // p = r + p
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
            // checkCudaErrors(cublasStatus);
        }
        else
        {
            // Initialize p = r
            // p = r
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
            // checkCudaErrors(cublasStatus);
        }

        // Ax = A * p
        // y = α op(A) * p + β * y
        cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, 
                N, N, 
                &alpha,
                d_Sigma, N,
                d_p, 1,
                &beta, 
                d_Ax, 1);
        // checkCudaErrors(cublasStatus);

        // dot = p^T * Ax = p^T * A * p
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        // checkCudaErrors(cublasStatus);
        a = r1 / dot;

        // x = a * p + x
        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        // checkCudaErrors(cublasStatus);
        na = -a;
        // r = - a * Ax = - a * A * p
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);
        // checkCudaErrors(cublasStatus);
        r0 = r1;
        // r1 = r^T * r
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        // checkCudaErrors(cublasStatus);

        cudaThreadSynchronize();
        // printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaFree(d_p);
    cudaFree(d_Ax);
}
#endif
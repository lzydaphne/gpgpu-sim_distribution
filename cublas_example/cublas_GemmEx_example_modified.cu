#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;

    // Use __half for A and B, float for C
    std::vector<__half> A = {__float2half(1.0), __float2half(3.0), __float2half(2.0), __float2half(4.0)};
    std::vector<__half> B = {__float2half(5.0), __float2half(7.0), __float2half(6.0), __float2half(8.0)};
    std::vector<float> C(m * n);
    
    // Use float for alpha and beta to match computation type
    // float alpha = 1.0;
    // float beta = 0.0;
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    __half *d_A = nullptr;
    __half *d_B = nullptr;
    float *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Initialize cuBLAS, CUDA
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // Allocate memory for A, B as __half, and C as float
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(__half) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(__half) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * C.size()));

    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(__half) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(__half) * B.size(), cudaMemcpyHostToDevice, stream));

    // Perform GEMM operation using cublasGemmEx
    CUBLAS_CHECK(cublasGemmEx(
        cublasH, 
        transa, transb, 
        m, n, k, 
        &alpha, 
        d_A, CUDA_R_16F, lda, 
        d_B, CUDA_R_16F, ldb, 
        &beta, 
        d_C, CUDA_R_32F, ldc, 
        CUBLAS_COMPUTE_32F_PEDANTIC, 
        CUBLAS_GEMM_DEFAULT
    ));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(float) * C.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

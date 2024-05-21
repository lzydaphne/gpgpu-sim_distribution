/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>


// Define some error checking macros.
#define cudaErrCheck(stat) \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}


#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>
using namespace nvcuda;

#define MATRIX_M 20480
#define MATRIX_N 20480
#define MATRIX_K 20480
#define THRESHOLD_ROW 20464
#define THRESHOLD_COL 20464
// #define MATRIX_M 10240
// #define MATRIX_N 10240
// #define MATRIX_K 10240
// #define THRESHOLD_ROW 10224
// #define THRESHOLD_COL 10224
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void sp_example(half *a, half *b, float *c, int M, int N, int K,
                           float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < THRESHOLD_ROW || col < THRESHOLD_COL) return;
  if (row >= M || col >= N) return;
 
  half sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    half a_val = a[row * K + k];  // a is MxK
    half b_val = b[k * N + col];  // b is KxN

    //* caculation down below perform slower
    sum = __hadd(sum, __hadd(__hmul(alpha, a_val), __hmul(beta, b_val)));

  }
c[row * N + col] += __half2float(sum);
}

// CD kernel
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration
// purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K,
                             float alpha, float beta) {

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  //! Only skip processing if both row and column indices exceed their thresholds
  if(warpM * WMMA_M >= THRESHOLD_ROW && warpN * WMMA_N >= THRESHOLD_COL) return;

  // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for (int i = 0; i < K; i += WMMA_K) {
    // printf("----start--target---sp--\n");
    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);

      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  // Ensure only the first thread in each warp prints

  if (cRow < M && cCol < N) {
   wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
    // printf("wmma_example: cRow = %d, cCol = %d\n", cRow, cCol);
    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];

    /// Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);

    } 
  }
}

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
  // count cycles

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

int main(int argc, char *argv[]) {
  // Initialize the problem
  int nkernels = 2;             // number of concurrent kernels
  int nstreams = nkernels + 1;  // use one more stream than concurrent kernel
  // int nbytes = nkernels * sizeof(float_t);  // number of data bytes
  float kernel_time = 10;             // time the kernel should run in ms
  float elapsed_time, elapsed_time2;  // timing variables
  int cuda_device = 0;

  cudaDeviceProp deviceProp;
  cudaErrCheck(cudaGetDevice(&cuda_device));

  cudaErrCheck(cudaGetDeviceProperties(&deviceProp, cuda_device));

  if ((deviceProp.concurrentKernels == 0)) {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  } else {
    printf("concurrent kernel: %d\n", deviceProp.concurrentKernels);
  }

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    //evaluate the performance
    cudaEvent_t start, stop;
    float elapsedTime;


  // cuda core
  int N = MATRIX_N;  // Define the size of the matrix
  size_t size = N * N * sizeof(float_t);
  float *h_A, *h_B, *h_C;           // host copies of A, B, C
  float *d_A, *d_B, *d_C;           // device copies of A, B, C

  // float *a_fp32;
  // float *b_fp32;
  half *a_fp16;
  half *b_fp16;
  // //printf("WMMA Example2\n");

  float *c;
  //    float *c_cublas;
  float *c_wmma;

  //    float *c_host_cublas;
  float *c_host_wmma;
  // //printf("WMMA Example3\n");

  // cuda core: Allocate space for host copies and setup values
  cudaErrCheck(cudaMallocHost((void **)&h_A, size));
  cudaErrCheck(cudaMallocHost((void **)&h_B, size));
  cudaErrCheck(cudaMallocHost((void **)&h_C, size));

  // Allocate space for device copies
  cudaErrCheck(cudaMalloc((void **)&d_A, size));
  cudaErrCheck(cudaMalloc((void **)&d_B, size));
  cudaErrCheck(cudaMalloc((void **)&d_C, size));
  cudaErrCheck(
      cudaMalloc((void **)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(
      cudaMalloc((void **)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

  // Initialize matrices A and B with random values
  for (int i = 0; i < N * N; i++) {
    h_A[i] = (float)rand() / (float)RAND_MAX *
             10.0;  // Assign a random float value between 0 and 100
    h_B[i] = (float)rand() / (float)RAND_MAX *
             10.0;  // Assign a random float value between 0 and 100
    h_C[i] = 1.5;
  }
  // stream create
  // allocate and initialize an array of stream handles
  cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

  cudaStreamCreateWithPriority(&streams[0], cudaStreamNonBlocking, 0);
  cudaStreamCreateWithPriority(&streams[1], cudaStreamNonBlocking, 1);
  cudaStreamCreateWithPriority(&streams[2], cudaStreamNonBlocking, 2);
  // for (int i = 1; i < nstreams; i++) {
  //   cudaErrCheck(cudaStreamCreate(&(streams[i])));
  // }

  // Copy inputs to device
  cudaErrCheck(
      cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, streams[0]));
  cudaErrCheck(
      cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, streams[1]));
  cudaErrCheck(
      cudaMemcpyAsync(d_C, h_C, size, cudaMemcpyHostToDevice, streams[2]));
  
  //! concurrent

  printf("Converting to fp16...a_fp16\n");
  convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 31) / 32, 32,0,0>>>(
      a_fp16, d_A, MATRIX_M * MATRIX_K);
  //! concurrent
  printf("Converting to fp16...b_fp16\n");
 
  convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 31) / 32, 32,0,0>>>(
      b_fp16, d_B, MATRIX_K * MATRIX_N);
 

  float alpha = 2.0f;
  float beta = 2.0f;
  half alpha_fp16 = 2.0;
  half beta_fp16 = 2.0;

  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;
 
  gridDim.x =
      (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);


//* Create events for timing
cudaEventCreate(&start);
cudaEventCreate(&stop);
 cudaEventRecord(start, 0);


//! wmma kernel
// wmma_example<<<gridDim, blockDim, 0, 0>>>(
//       a_fp16, b_fp16, d_C, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
wmma_example<<<gridDim, blockDim, 0, streams[0]>>>(
      a_fp16, b_fp16, d_C, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);


  dim3 sp_blockDim(16,16);  // Commonly used block size for matrix multiplication
  dim3 sp_gridDim;
  sp_gridDim.x = (N + sp_blockDim.x - 1) / sp_blockDim.x;
  sp_gridDim.y = (N + sp_blockDim.y - 1) / sp_blockDim.y;
  printf("sp_example's : gridDim.x = %d, gridDim.y = %d\n", sp_gridDim.x,
         sp_gridDim.y);

//! sp kernel
sp_example<<<sp_gridDim, sp_blockDim, 0, streams[1]>>>(
        a_fp16, b_fp16, d_C, MATRIX_M, MATRIX_N, MATRIX_K, alpha_fp16, beta_fp16);
    
  

  cudaStreamSynchronize(streams[0]);
  cudaStreamSynchronize(streams[1]);


cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime, start, stop);
printf("Time for WMMA kernel: %f ms\n", elapsedTime);
cudaEventDestroy(start);
cudaEventDestroy(stop);

 //* use to prevent segfault!!!
  cudaDeviceSynchronize();

  // Error checking
  printf("\nChecking results...\n");

  cudaErrCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));


  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  cudaStreamDestroy(streams[2]);
  free(streams);

  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));


  // cuda core
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  // cudaFree(d_tmpC);
  //* Pinned Memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);

  exit(EXIT_SUCCESS);
}
#include <iostream>
// System includes
#include <stdio.h>
#include <assert.h>
// #include <curand.h>
// #include <cublas_v2.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
using namespace std;
//    N         K        K 
// M  A   x   N B  =   M C

#define M 32*32
#define N 32*32
#define K 32*32

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

__global__ void launch_kernel(int *A, int *B, int *C){
        __shared__ int tile_a[BLOCKSIZE_Y][BLOCKSIZE_X];
        __shared__ int tile_b[BLOCKSIZE_Y][BLOCKSIZE_X];

        const unsigned int C_block_row = blockIdx.y;
        const unsigned int C_block_col = blockIdx.x;
        const unsigned int A_block_row = C_block_row;
        unsigned int A_block_col;
        unsigned int B_block_row;
        const unsigned int B_block_col = C_block_col;
        unsigned int A_id,B_id,C_id,sum ;
        const unsigned int C_thread_row = threadIdx.y;
        const unsigned int C_thread_col = threadIdx.x;
        const unsigned int A_thread_row = C_thread_row;
        unsigned int A_thread_col;
        unsigned int B_thread_row;
        const unsigned int B_thread_col = C_thread_col;


        sum=0;

        for(int i=0;  i < N; i ++){
                A_block_col = i;
                B_block_row = i;
                A_id = N * (blockDim.y * A_block_row + threadIdx.y)  + A_block_col * blockDim.x  + threadIdx.x;
                B_id = K * (blockDim.y * B_block_row + threadIdx.y)  + B_block_col * blockDim.x  + threadIdx.x;
                tile_a[threadIdx.y][threadIdx.x] = A[A_id];
                tile_b[threadIdx.y][threadIdx.x] = B[B_id];
                __syncthreads();

                for(int j=0; j<blockDim.x ; j++){
                        A_thread_col = j;
                        B_thread_row = j;
                        sum += tile_a[A_thread_row][A_thread_col] * tile_b[B_thread_row][B_thread_col];
                }

        }
        C_id = K * (blockDim.y * C_block_row + C_thread_row)  + C_block_col * blockDim.x  + C_thread_col;
        C[C_id] = sum;


}

int main(){
        int* A_h = new int[M*N];
        int* B_h = new int[N*K];
        int* C_h = new int[M*K];
        cout << "Matrix A:" <<endl;
        for(int i=0 ; i < M; i++){
                for(int j=0 ; j < N; j++){
                        A_h[i*M+j] = rand()%5;
                        cout << A_h[i*M + j] << "\t" ;
                }
                cout<<endl;
        }

        cout << "Matrix B:" <<endl;
        for(int i=0 ; i < N; i++){
                for(int j=0 ; j < K; j++){
                        B_h[i*N+j] = rand()%5;
                        cout << B_h[i*N + j] << "\t" ;
                }
                cout<<endl;
        }


        int *A_d, *B_d, *C_d;
        cudaMalloc(&A_d, M*N*sizeof(int));
        cudaMalloc(&B_d, N*K*sizeof(int));
        cudaMalloc(&C_d, M*K*sizeof(int));
        cudaMemcpy(A_d,A_h,M*N*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(B_d,B_h,N*K*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(C_d,C_h,M*K*sizeof(int),cudaMemcpyHostToDevice);

        dim3 block(BLOCKSIZE_X,BLOCKSIZE_Y,1);
        dim3 grid((K + BLOCKSIZE_X -1) /BLOCKSIZE_X, (M + BLOCKSIZE_Y -1) / BLOCKSIZE_Y);

        launch_kernel<<<grid,block>>>(A_d,B_d,C_d);
        cudaDeviceSynchronize();
        cudaMemcpy(C_h,C_d,M*K*sizeof(int),cudaMemcpyDeviceToHost);

        cout << "Matrix C:" <<endl;
        for(int i=0 ; i < M; i++){
                for(int j=0 ; j < K; j++)
                        cout << C_h[i*K + j] << "\t" ;
                cout << endl;
        }

        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        delete [] A_h;
        delete [] B_h;
        delete [] C_h;

}
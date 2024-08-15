#include <stdio.h>
#include <cuda_runtime.h>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
        exit(1);
    }
}

__constant__ float predefined_input[6][6] = {
    { 3.14f,  -2.71f,  8.90f,  -5.23f,  12.45f, -7.89f},
    { 0.00f,  14.70f,  0.00f,  18.30f,  -1.50f,  9.80f},
    { 0.57f, -22.30f, 13.80f,  -7.10f,  25.60f, -3.90f},
    {-15.30f,  7.80f, -11.20f, 23.90f,  -6.40f, 17.50f},
    { 4.68f,  -9.10f, 27.30f,  -2.40f,  14.90f, -7.60f},
    {-12.60f,  5.30f, -20.10f,  9.70f,  -3.80f, 28.50f}
};

__constant__ float conv_kernel3[3][3] = {
    {1.0f, 0.0f, -1.0f},
    {2.0f, 0.0f, -2.0f},
    {1.0f, 0.0f, -1.0f}
};

__global__ void convolution_2D_basic_kernel(float *P, int input_width, int input_height, int output_width, int output_height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (outCol < output_width && outRow < output_height) {
        float Pvalue = 0.0f;
        int r = 1;  // Radius of the 3x3 kernel
       
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                int inRow = outRow + i + r;
                int inCol = outCol + j + r;
                if (inRow >= 0 && inRow < input_height && inCol >= 0 && inCol < input_width) {
                    Pvalue += conv_kernel3[i+r][j+r] * predefined_input[inRow][inCol];
                }
            }
        }
        P[outRow * output_width + outCol] = Pvalue;
    }
}

int main() {
    int input_width = 6, input_height = 6;
    int kernel_size = 3;  // 3x3 kernel
    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    size_t output_size = output_width * output_height * sizeof(float);

    float *h_output = (float *)malloc(output_size);

    // Allocate device memory
    float *d_output;
    cudaErrCheck(cudaMalloc((void **)&d_output, output_size));

    // Define block and grid dimensions
    dim3 blockDim(2, 2);  // Adjusted for 6x6 input
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x,
                 (output_height + blockDim.y - 1) / blockDim.y);

    // Print block and grid dimensions
    printf("blockDim: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("gridDim: (%d, %d)\n", gridDim.x, gridDim.y);

    // Launch convolution kernel
    convolution_2D_basic_kernel<<<gridDim, blockDim>>>(d_output, input_width, input_height, output_width, output_height);
    cudaErrCheck(cudaGetLastError());
    cudaErrCheck(cudaDeviceSynchronize());

    // Copy result back to host
    cudaErrCheck(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Print convolution results
    printf("\nConvolution Results:\n");
    printf("output_height: %d, output_width: %d\n", output_height, output_width);
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            printf("%f ", h_output[i * output_width + j]);
        }
        printf("\n");
    }

    // Free memory
    free(h_output);
    cudaErrCheck(cudaFree(d_output));

    return 0;
}
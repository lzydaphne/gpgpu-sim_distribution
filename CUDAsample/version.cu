#include <iostream>
#include <cuda_runtime_api.h>

int main() {
    std::cout << "CUDART_VERSION: " << CUDART_VERSION << std::endl;
    std::cout << "Compiled against CUDA Runtime version: "
              << CUDART_VERSION / 1000 << "." << (CUDART_VERSION % 1000) / 10 << std::endl;
    return 0;
}

#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();

    return 0;
}

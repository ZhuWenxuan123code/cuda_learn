#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    int device = 0;  // GPU 编号，从 0 开始

    cudaDeviceProp prop{};
    cudaError_t error = cudaGetDeviceProperties(&prop, device);

    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("Device: %s\n", prop.name);
    printf("regsPerMultiprocessor = %d\n", prop.regsPerMultiprocessor);
    printf("regsPerBlock = %d\n", prop.regsPerBlock);

    printf("maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
    return 0;
}

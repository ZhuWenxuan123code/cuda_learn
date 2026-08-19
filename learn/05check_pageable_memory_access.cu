#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA device found.\n");
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        int pageableMemoryAccess = 0;
        cudaDeviceGetAttribute(
            &pageableMemoryAccess,
            cudaDevAttrPageableMemoryAccess,
            device);

        printf("Device %d: %s\n", device, prop.name);
        printf("  cudaDevAttrPageableMemoryAccess = %d\n",
               pageableMemoryAccess);

        if (pageableMemoryAccess == 1) {
            printf("  Result: This system supports direct GPU access to\n");
            printf("          ordinary malloc/new/mmap memory.\n");
            printf("          All host memory effectively behaves as\n");
            printf("          unified memory.\n");
        } else {
            printf("  Result: Ordinary malloc/new/mmap memory is NOT\n");
            printf("          automatically unified memory.\n");
            printf("          You must use cudaMallocManaged() or other\n");
            printf("          explicit managed allocations.\n");
        }

        printf("\n");
    }

    return 0;
}

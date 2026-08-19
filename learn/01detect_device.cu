#include <cuda_runtime.h>
#include <iostream>

// 同一个函数可以被 CPU 和 GPU 调用。
__host__ __device__ int getValue()
{
#ifdef __CUDA_ARCH__
    // 编译 GPU 版本时，__CUDA_ARCH__ 已定义。
    return threadIdx.x;
#else
    // 编译 CPU 版本时，__CUDA_ARCH__ 未定义。
    return 0;
#endif
}

__global__ void testKernel(int* values)
{
    values[threadIdx.x] = getValue();
}

int main()
{
    constexpr int threadCount = 7;
    int* values = nullptr;

    cudaMallocManaged(&values, threadCount * sizeof(int));

    // 这里调用的是 getValue() 的 CPU 版本，因此返回 0。
    std::cout << "CPU value: " << getValue() << '\n';

    // 这里调用的是 getValue() 的 GPU 版本。
    testKernel<<<1, threadCount>>>(values);
    cudaDeviceSynchronize();

    std::cout << "GPU values: ";
    for (int i = 0; i < threadCount; ++i) {
        std::cout << values[i] << ' ';
    }
    std::cout << '\n';

    cudaFree(values);
    return 0;
}

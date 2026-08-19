#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
// #include <cstdlib>
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 10;
    const size_t bytes = N * sizeof(float);

    // 1. Host Memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = 2 * static_cast<float>(i);
    }

    // 2. Device Memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. Host to Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // error checking
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr
            << "Kernel launch failed: "
            << cudaGetErrorString(err)
            << '\n';
    }

    // 5. Device to Host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    bool correct = true;
    // 6. Verify
    for (int i = 0; i < N; i++)
    {
        float expected = h_A[i] + h_B[i];
        // std::cout << expected << h_C[i] << std::endl;
        if (std::fabs(expected - h_C[i]) > 1e-5)
        {
            std::cout << "Mismatch at " << i << ": expected " << expected << "; got " << h_C[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct)
    {
        std::cout << "Vector add correct!\n";
    }

    // 7. free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}

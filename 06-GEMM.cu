#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#define CHECK_CUDA(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cudaError_t err = call;                                                                                 \
        if (err != cudaSuccess)                                                                                 \
        {                                                                                                       \
            std::cerr << "CUDA error at :" << __FILE__ << ":" << __LINE__ << "-" << cudaGetErrorString << "\n"; \
            std::exit(EXIT_FAILURE);                                                                            \
        }                                                                                                       \
    } while (0)

constexpr int TILE_DIM = 16;
/*
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]
 *
 * row-major:
 *
 * A[row, col] = A[row * K + col]
 * B[row, col] = B[row * N + col]
 * C[row, col] = C[row * N + col]
 */

// =====================================================
// CPU Reference GEMM
// =====================================================

void cpu_gemm(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}
// =====================================================
// Naive CUDA GEMM
//
// One thread computes one C element.
// =====================================================

__global__ void gemm_naive_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {

        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            float a = A[row * K + k];
            float b = B[k * N + col];
            sum += a * b;
        }

        C[row * N + col] = sum;
    }
}
// =====================================================
// Shared Memory Tiled GEMM
//
// One block computes one TILE_DIM x TILE_DIM tile of C.
// One thread computes one C element.
// =====================================================
__global__ void gemm_shared_kernel(const float *A,
                                   const float *B,
                                   float *C,
                                   int M,
                                   int N,
                                   int K)
{
    __shared__ float As[TILE_DIM][TILE_DIM]; // 不会出现bank conflict
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // C
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // C

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    /*
     * K 维度被切成多个 tile。
     *
     * 每一轮 phase:
     *
     * A tile: [row block, phase]
     * B tile: [phase, col block]
     */

    for (int t = 0; t < K; t += TILE_DIM)
    {
        // As[ty][tx] = A[row * K + (t + tx)]; // 没有零填充保护
        // Bs[ty][tx] = B[(t + ty) * N + col]; // 没有零填充保护
        As[ty][tx] = (row < M && t + tx < K)
                         ? A[row * K + (t + tx)]
                         : 0.0f;

        Bs[ty][tx] = (t + ty < K && col < N)
                         ? B[(t + ty) * N + col]
                         : 0.0f;
        __syncthreads();                   // 等大家都搬完
        for (int k = 0; k < TILE_DIM; k++) // 只在 Shared MEM 里算
        {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}
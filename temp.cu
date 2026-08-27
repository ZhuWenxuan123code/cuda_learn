#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                       \
    do                                                         \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess)                                \
        {                                                      \
            std::cerr << cudaGetErrorString(err) << std::endl; \
            exit(1);                                           \
        }                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;

// -----------------------------
// Warp Reduce Sum
// -----------------------------

__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = 16;
         offset > 0;
         offset >>= 1)
    {
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset);
    }

    return val;
}

// -----------------------------
// Block Reduce Sum
// -----------------------------

__device__ float blockReduceSum(float val)
{
    __shared__ float shared[32];

    int lane =
        threadIdx.x % 32;

    int warp =
        threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0)
    {
        shared[warp] = val;
    }

    __syncthreads();

    int num_warps =
        (blockDim.x + 31) / 32;

    val =
        (threadIdx.x < num_warps)
            ? shared[lane]
            : 0.0f;

    if (warp == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

// =================================================
// LayerNorm Kernel
//
// input:
//      [rows, hidden]
//
// gamma:
//      [hidden]
//
// beta:
//      [hidden]
//
// output:
//      [rows, hidden]
//
// one block = one row
// =================================================

__global__ void layernorm_kernel(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    int hidden,
    float eps)
{

    int row = blockIdx.x;

    const float *row_input =
        input + row * hidden;

    float *row_output =
        output + row * hidden;

    int tid = threadIdx.x;

    // -----------------------------
    // Step1:
    // calculate mean
    // -----------------------------

    float sum = 0.0f;

    for (int i = tid;
         i < hidden;
         i += blockDim.x)
    {
        sum += row_input[i];
    }

    float mean =
        blockReduceSum(sum) / hidden;

    // -----------------------------
    // Step2:
    // calculate variance
    // -----------------------------

    float var_sum = 0.0f;

    for (int i = tid;
         i < hidden;
         i += blockDim.x)
    {
        float diff =
            row_input[i] - mean;

        var_sum += diff * diff;
    }

    float variance =
        blockReduceSum(var_sum) / hidden;

    float inv_std =
        rsqrtf(
            variance + eps);

    // -----------------------------
    // Step3:
    // normalize
    // -----------------------------

    for (int i = tid;
         i < hidden;
         i += blockDim.x)
    {

        float norm =
            (row_input[i] - mean) *
            inv_std;

        row_output[i] =
            norm * gamma[i] +
            beta[i];
    }
}

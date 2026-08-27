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
        }                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;
// -----------------------------
// Warp Reduce Sum
// -----------------------------
__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -----------------------------
// Block Reduce Sum
// -----------------------------
__device__ float blockReduceSum(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    val = warpReduceSum(val); // 计算每个warp的 sum val
    if (lane == 0)
        shared[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0)
    {
        val = lane < num_warps ? shared[lane] : 0.0f;
        val = warpReduceSum(val); // 计算单个warp的sum val
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
__global__ void layernorm_kernel(const float *input, const float *gamma, const float *beta, float *output, int hidden, float eps)
{
    int row = blockIdx.x;
    const float *row_input = input + row * hidden;
    float *row_output = output + row * hidden;

    int tid = threadIdx.x;
    // -----------------------------
    // Step1:
    // calculate mean
    // -----------------------------
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x)
    {
        sum += row_input[i];
    }
    // __syncthreads(); 为什么不用加？
    sum = blockReduceSum(sum);
    float mean = sum / hidden;

    // -----------------------------
    // Step2:
    // calculate variance
    // -----------------------------

    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x)
    {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = blockReduceSum(var_sum) / hidden;

    float inv_std = rsqrtf(variance + eps);
    // -----------------------------
    // Step3:
    // normalize
    // -----------------------------

    for (int i = tid; i < hidden; i += blockDim.x)
    {
        float norm = (row_input[i] - mean) * variance;
        row_output[i] = norm * gamma[i] + beta[i]; // 可学习参数 gamma 和beta
    }
}

int main()
{

    int rows = 4;
    int hidden = 1024;

    size_t size =
        rows * hidden * sizeof(float);

    std::vector<float> h_input(
        rows * hidden);

    std::vector<float> h_output(
        rows * hidden);

    std::vector<float> h_gamma(hidden);
    std::vector<float> h_beta(hidden);

    for (int i = 0; i < rows * hidden; i++)
    {
        h_input[i] = i % 100;
    }

    for (int i = 0; i < hidden; i++)
    {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    float *d_input;
    float *d_output;
    float *d_gamma;
    float *d_beta;

    CHECK_CUDA(cudaMalloc(
        &d_input,
        size));

    CHECK_CUDA(cudaMalloc(
        &d_output,
        size));

    CHECK_CUDA(cudaMalloc(
        &d_gamma,
        hidden * sizeof(float)));

    CHECK_CUDA(cudaMalloc(
        &d_beta,
        hidden * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        size,
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_gamma,
        h_gamma.data(),
        hidden * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_beta,
        h_beta.data(),
        hidden * sizeof(float),
        cudaMemcpyHostToDevice));

    layernorm_kernel<<<rows, BLOCK_SIZE>>>(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        hidden,
        1e-5);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_output.data(),
        d_output,
        size,
        cudaMemcpyDeviceToHost));

    std::cout
        << "LayerNorm finished\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}
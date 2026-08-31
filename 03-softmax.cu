#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t err = call;                               \
        if (err != cudaSuccess)                               \
        {                                                     \
            std::cerr << "CUDA error at: " << __FILE__ << ":" \
                      << __LINE__ << "-"                      \
                      << cudaGetErrorString(err) << "\n";     \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

constexpr int BLOCK_SIZE = 256;
// warp 内 max reduction
__device__ __forceinline__ float warpReduceMax(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}
// warp 内 sum reduction
__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block 内 max reduction
__device__ __forceinline__ float blockReduceMax(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    val = warpReduceMax(val);
    if (lane == 0)
        shared[warp_id] = val;
    __syncthreads();

    // 啥意思？？？？？？
    // 让 warp 0 来进行最后的规约。
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    // if (warp_id == 0)
    // {
    //     val = warpReduceMax(val);
    // }

    if (warp_id == 0) // 更易读的思路，此处if不能去掉，不然block里每个warp都会重复执行操作。
    {
        val = (lane < num_warps) ? shared[lane] : -INFINITY;
        val = warpReduceMax(val);

        if (lane == 0)
            shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}
// block 内 sum reduction
__device__ __forceinline__ float blockReduceSum(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0)
        shared[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0)
    {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
        if (lane == 0)
            shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}
/*
 * Row-wise softmax
 *
 * input:  [num_rows, num_cols]
 * output: [num_rows, num_cols]
 *
 * 一个 block 负责一行。
 */
__global__ void softmax_kernel(const float *input, float *output, int num_rows, int num_cols)
{
    int row = blockIdx.x;
    if (row >= num_rows)
        return;
    const float *row_input = input + row * num_cols;
    float *row_output = output + row * num_cols;
    int tid = threadIdx.x;
    // ------------------------------------
    // Step 1: 求当前行的最大值
    // ------------------------------------
    float local_max = -INFINITY;
    for (int col = tid; col < num_cols; col += blockDim.x) // 遍历列
    {
        local_max = fmaxf(local_max, row_input[col]); // local_max 不是“当前整行的最大值”，而是“当前这个线程负责的那些列的最大值”!!!
    }
    float row_max = blockReduceMax(local_max);
    // ------------------------------------
    // Step 2: 计算 exp(x - max)，同时求和
    // ------------------------------------
    float local_sum = 0.0f;
    for (int col = tid; col < num_cols; col += blockDim.x)
    {
        float val = expf(row_input[col] - row_max);
        row_output[col] = val;
        local_sum += val;
    }
    float row_sum = blockReduceSum(local_sum);

    // ------------------------------------
    // Step 3: 归一化
    // ------------------------------------

    for (int col = tid; col < num_cols; col += blockDim.x)
    {
        row_output[col] = row_output[col] / row_sum;
    }
}
// cpu softmax
void cpu_softmax(const std::vector<float> &input, std::vector<float> &output, int num_rows, int num_cols)
{
    for (int r = 0; r < num_rows; r++)
    {
        const float *row_input = input.data() + r * num_cols;
        float *row_output = output.data() + r * num_cols;
        float max = -INFINITY;
        for (int c = 0; c < num_cols; c++)
            max = fmaxf(max, row_input[c]);
        float sum = 0.0f;
        for (int c = 0; c < num_cols; c++)
        {
            row_output[c] = expf(row_input[c] - max);
            sum += row_output[c];
        }
        for (int c = 0; c < num_cols; c++)
            row_output[c] /= sum;
    }
}
int main()
{
    const int num_rows = 4;
    const int num_cols = 1024;

    const int num_elements = num_rows * num_cols;
    const size_t bytes = num_elements * sizeof(float);

    std::vector<float> h_input(num_elements);
    std::vector<float> h_output(num_elements);
    std::vector<float> h_ref(num_elements);

    for (int i = 0; i < num_elements; ++i)
    {
        h_input[i] = static_cast<float>((i % 17) - 8);
    }

    // CPU reference
    cpu_softmax(h_input, h_ref, num_rows, num_cols);

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice));

    dim3 grid(num_rows);
    dim3 block(BLOCK_SIZE);

    softmax_kernel<<<grid, block>>>(
        d_input,
        d_output,
        num_rows,
        num_cols);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost));

    // Verify
    float max_error = 0.0f;

    for (int i = 0; i < num_elements; ++i)
    {
        float err = std::fabs(h_output[i] - h_ref[i]);
        max_error = std::max(max_error, err);
    }

    std::cout << "Max error = " << max_error << "\n";

    for (int r = 0; r < num_rows; ++r)
    {
        float row_sum = 0.0f;

        for (int c = 0; c < num_cols; ++c)
        {
            row_sum += h_output[r * num_cols + c];
        }

        std::cout << "Row " << r << " sum = " << row_sum << "\n";
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}

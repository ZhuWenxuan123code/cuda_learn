#include <cuda_runtime.h>
#include <iostream>
constexpr int BLOCK_SIZE = 256;

__global__ void reduction_sum(const float *input, float *output, int N)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // global memory --> shared memory
    if (idx < N)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Tree Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

/*
 * warp 内归约
 *
 * 每次让当前 lane 和后面 offset 个 lane 的值相加：
 *
 * offset = 16
 * offset = 8
 * offset = 4
 * offset = 2
 * offset = 1
 *
 * 最终 lane 0 得到整个 warp 的 sum
 */
__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = warpSize / 2;
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
__global__ void reduction_sum_optimized(const float *input, float *output, int N)
{
    /*
     * 一个 block 最多有 1024 threads
     * = 32 warps
     *
     * 所以最多需要保存 32 个 warp partial sums。
     */
    __shared__ float warpSums[32];
    const int tid = threadIdx.x;
    const int laneId = tid % 32;
    const int warpId = tid / 32;
    /*
     * 一个 block 现在负责：
     *
     * 2 * blockDim.x
     *
     * 个元素。
     */
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // 跨block增加
    // --------------------------------
    // Step 1:
    // 每个线程加载两个元素
    // 并直接在 register 中相加
    // --------------------------------
    float sum = 0.0f;
    if (idx < N)
    {
        sum += input[idx];
    }
    if (idx + blockDim.x < N)
    {
        sum += input[idx + blockDim.x];
    }

    // --------------------------------
    // Step 2:
    // 每个 warp 内进行 reduction
    // --------------------------------
    sum = warpReduceSum(sum);

    // --------------------------------
    // Step 3:
    // 每个 warp 的 lane 0
    // 把 warp sum 写入 shared memory
    // --------------------------------
    if (laneId == 0)
    {
        warpSums[warpId] = sum;
    }

    /*
     * 必须同步：
     *
     * 因为下面 warp 0
     * 要读取其他 warp 写入的 warpSums[]
     */
    __syncthreads();
    // --------------------------------
    // Step 4:
    // 让 warp 0 对所有 warp sum
    // 再做一次 reduction
    // --------------------------------
    if (warpId == 0)
    {

        const int numWarps =
            (blockDim.x + warpSize - 1) / warpSize;

        /*
         * 假设 blockDim.x = 256：
         *
         * 一共 8 个 warp
         *
         * lane0 -> warpSums[0]
         * lane1 -> warpSums[1]
         * ...
         * lane7 -> warpSums[7]
         *
         * lane8~31 -> 0
         */
        float blockSum =
            (laneId < numWarps)
                ? warpSums[laneId]
                : 0.0f;

        blockSum =
            warpReduceSum(blockSum);

        // --------------------------------
        // Step 5:
        // 整个 block 的最终结果
        // --------------------------------

        if (laneId == 0)
        {
            output[blockIdx.x] = blockSum;
        }
    }
}
int main()
{
    const int N = 10086;
    const size_t bytes = N * sizeof(float);
    float *h_input = new float[N];

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    float *d_input = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // int gridSize = (N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE); // optimized

    float *d_output = nullptr;
    cudaMalloc(&d_output, gridSize * sizeof(float));

    reduction_sum<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);

    // error catch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel Launch Failed: "
                  << cudaGetErrorString(err)
                  << "\n";
        return 1;
    }

    float *h_output = new float[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float result = 0;
    for (int i = 0; i < gridSize; i++)
    {
        result += h_output[i];
    }
    std::cout << "reduction sum:" << result << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
}
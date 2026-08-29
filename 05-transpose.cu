#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                     \
    do                                                       \
    {                                                        \
        cudaError_t err = call;                              \
        if (err != cudaSuccess)                              \
        {                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" \
                      << __LINE__ << " - "                   \
                      << cudaGetErrorString(err) << "\n";    \
            std::exit(EXIT_FAILURE);                         \
        }                                                    \
    } while (0)

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// 每个线程负责处理一个元素
__global__ void transpose_naive_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < cols && y < rows)
    {
        output[x * rows + y] = input[y * cols + x];
    }
}

__global__ void transpose_naive_kernel(const float *input, float *output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col of input
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row of input
    if (x < rows && y < cols)
    {
        output[x * rows + y] = input[y * cols + x];
    }
}

// =====================================================
// Shared memory optimized transpose
//
// input:
//      A[rows, cols]
//
// output:
//      C[cols, rows]
//
// 一个 block 处理一个 32 x 32 tile。
// 使用 tile[32][33] 避免 shared memory bank conflict。
// =====================================================

__global__ void transpose_shared_kernel(const float *input, float *output, int rows, int cols)
{
    /*
     * +1 padding:
     *
     * tile[32][32] 在转置访问时容易产生 32-way bank conflict。
     * tile[32][33] 让每一行跨度从 32 float 变成 33 float，
     * 从而打散 bank 映射。
     */
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    /*
     * 当前 block 负责 input 中的一个 tile:
     *
     * x: input 的列坐标
     * y: input 的行坐标
     */
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    /*
     * Step 1:
     * 从 global memory 读取 input tile 到 shared memory。
     *
     * BLOCK_ROWS = 8，因此一个 32x8 的 block 总共 256 threads。
     * 每个线程负责 4 个元素：
     *
     * j = 0, 8, 16, 24
     */
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int yy = y + j; // input中的绝对坐标
        if (x < cols && yy < rows)
        {
            tile[threadIdx.y + j][threadIdx.x] = input[yy * cols + x];
        }
    }
    __syncthreads();
    /*
     * Step 2:
     * 写回 output。
     *
     * 注意 blockIdx.x 和 blockIdx.y 交换了角色。
     *
     * input tile:
     *      rows direction: blockIdx.y
     *      cols direction: blockIdx.x
     *
     * output tile:
     *      rows direction: blockIdx.x
     *      cols direction: blockIdx.y
     */
    x = blockIdx.y * TILE_DIM + threadIdx.x; // output col, corresponds to input row
    y = blockIdx.x * TILE_DIM + threadIdx.y; // output row, corresponds to input col

    /*
     * output 的形状是 [cols, rows]
     *
     * output row 范围: 0 ~ cols-1
     * output col 范围: 0 ~ rows-1
     */

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int yy = y + j; // 绝对 output row
        if (x < rows && yy < cols)
        {
            output[yy * rows + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// =====================================================
// Verify
// =====================================================

bool verify(
    const std::vector<float> &a,
    const std::vector<float> &b,
    float eps = 1e-5f)
{
    if (a.size() != b.size())
    {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::fabs(a[i] - b[i]) > eps)
        {
            std::cerr << "Mismatch at " << i
                      << ": ref = " << a[i]
                      << ", got = " << b[i]
                      << "\n";
            return false;
        }
    }

    return true;
}

void cpu_transpose(
    const std::vector<float> &input,
    std::vector<float> &output,
    int rows,
    int cols)
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

// =====================================================
// Main
// =====================================================

int main()
{
    /*
     * 故意设置成非 32 整数倍，测试边界判断是否正确。
     */
    const int rows = 1023;
    const int cols = 2049;

    const int input_numel = rows * cols;
    const int output_numel = cols * rows;

    const size_t input_bytes = input_numel * sizeof(float);
    const size_t output_bytes = output_numel * sizeof(float);

    std::vector<float> h_input(input_numel);
    std::vector<float> h_output_naive(output_numel);
    std::vector<float> h_output_shared(output_numel);
    std::vector<float> h_ref(output_numel);

    for (int i = 0; i < input_numel; ++i)
    {
        h_input[i] = static_cast<float>(i % 1000) * 0.001f;
    }

    cpu_transpose(h_input, h_ref, rows, cols);

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        input_bytes,
        cudaMemcpyHostToDevice));

    // =================================================
    // Naive transpose
    // =================================================

    {
        dim3 block(32, 8);

        dim3 grid(
            (cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);

        CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

        transpose_naive_kernel<<<grid, block>>>(
            d_input,
            d_output,
            rows,
            cols);

        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(
            h_output_naive.data(),
            d_output,
            output_bytes,
            cudaMemcpyDeviceToHost));

        bool ok = verify(h_ref, h_output_naive);

        std::cout << "Naive transpose: "
                  << (ok ? "PASSED" : "FAILED")
                  << "\n";
    }

    // =================================================
    // Shared memory transpose
    // =================================================

    {
        dim3 block(TILE_DIM, BLOCK_ROWS);

        dim3 grid(
            (cols + TILE_DIM - 1) / TILE_DIM,
            (rows + TILE_DIM - 1) / TILE_DIM);

        CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

        transpose_shared_kernel<<<grid, block>>>(
            d_input,
            d_output,
            rows,
            cols);

        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(
            h_output_shared.data(),
            d_output,
            output_bytes,
            cudaMemcpyDeviceToHost));

        bool ok = verify(h_ref, h_output_shared);

        std::cout << "Shared memory transpose: "
                  << (ok ? "PASSED" : "FAILED")
                  << "\n";
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
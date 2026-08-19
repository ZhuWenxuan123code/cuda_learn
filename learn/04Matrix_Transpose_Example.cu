#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define INDX(row, col, ld) (((row) * (ld)) + (col))
/* macro to index a 1D memory array with 2D indices in row-major order */
/* ld is the leading dimension, i.e. the number of columns in the matrix*/

/* CUDA kernel for naive matrix transpose */
// 原始转置 kernel 的写操作效率低下，是因为写入时地址跨度太大（strided writes）
__global__ void naive_cuda_transpose(int m, float *a, float *c )
{
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    if( myRow < m && myCol < m )
    {
        c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
    } /* end if */
    return;
} /* end naive_cuda_transpose */

/* CUDA kernel for shared memory matrix transpose */

__global__ void smem_cuda_transpose(int m, float *a, float *c )
{

    /* declare a statically allocated shared memory array */

    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    /* determine my row tile and column tile index */

    const int tileCol = blockDim.x * blockIdx.x;
    const int tileRow = blockDim.y * blockIdx.y;

    /* read from global memory into shared memory array */
    smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];

    /* synchronize the threads in the thread block */
    __syncthreads();

    /* write the result from shared memory to global memory */
    c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];
    return;

} /* end smem_cuda_transpose */

int main()
{
    const int m = 5;
    const int elementCount = m * m;
    const size_t bytes = elementCount * sizeof(float);

    std::vector<float> h_a(elementCount);
    std::vector<float> h_c(elementCount, 0.0f);

    // 初始化一个 m x m 的矩阵。
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            h_a[INDX(row, col, m)] = static_cast<float>(row * 10 + col);
        }
    }

    float* d_a = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

    // 每个线程负责矩阵中的一个元素。
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (m + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_cuda_transpose<<<blocksPerGrid, threadsPerBlock>>>(m, d_a, d_c);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(d_a);
        cudaFree(d_c);
        return 1;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Kernel execution failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(d_a);
        cudaFree(d_c);
        return 1;
    }

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Original matrix:\n";
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            std::cout << h_a[INDX(row, col, m)] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nTransposed matrix:\n";
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            std::cout << h_c[INDX(row, col, m)] << ' ';
        }
        std::cout << '\n';
    }

    bool correct = true;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            if (h_c[INDX(col, row, m)] != h_a[INDX(row, col, m)]) {
                correct = false;
            }
        }
    }

    std::cout << "\nTranspose result: "
              << (correct ? "PASS" : "FAIL") << '\n';

    cudaFree(d_a);
    cudaFree(d_c);
    return correct ? 0 : 1;
}

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_shared_kernel(const float *input, float *output, int rows, int cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // col
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // row

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int yy = y + j;
        if (x < cols && yy < rows)
            tile[threadIdx.y + j][threadIdx.x] = input[yy * cols + x];
    }
    __syncthreads(); // 同步。

    int x = blockIdx.y * TILE_DIM + threadIdx.x; // col
    int y = blockIdx.x * TILE_DIM + threadIdx.y; // row

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int yy = y + j;
        if (x < rows && yy < cols)
        {
            output[yy * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
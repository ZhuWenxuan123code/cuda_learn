## 设计知识点
- 多线程协作
- __shared__
- __syncthreads()
- 树形归约
- block 内 partial sum
- 多 block 结果汇总

## reduction-sum
cpu写法
``` cpp
float sum = 0;

for (int i = 0; i < N; ++i) {
    sum += A[i];
}
```
CUDA 的难点是：
> 很多线程可以同时读数据，但最后必须合并成一个结果。
> Tree Reduction，树形归约。

## 最经典的 reduction kernel
``` cpp
__global__
void reduce_sum(
    const float* input,
    float* output,
    int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

## 把这个 reduction 从教学版本优化成 2 elements/thread + warp shuffle 版本。
- [ ]  还没看懂，回头再看  

优化 1：一个线程加载两个元素
优化 2：Warp-level reduction
优化 3：递归调用 reduction kernel
``` cpp
constexpr int BLOCK_SIZE = 256;


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
__device__ __forceinline__
float warpReduceSum(float val)
{
    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1)
    {
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );
    }

    return val;
}


__global__
void reduce_sum_optimized(
    const float* input,
    float* output,
    int N)
{
    /*
     * 一个 block 最多有 1024 threads
     * = 32 warps
     *
     * 所以最多需要保存 32 个 warp partial sums。
     */
    __shared__ float warpSums[32];


    const int tid = threadIdx.x;

    const int laneId = tid % warpSize;
    const int warpId = tid / warpSize;


    /*
     * 一个 block 现在负责：
     *
     * 2 * blockDim.x
     *
     * 个元素。
     */
    int idx =
        blockIdx.x * (blockDim.x * 2)
        + threadIdx.x;


    // --------------------------------
    // Step 1:
    // 每个线程加载两个元素
    // 并直接在 register 中相加
    // --------------------------------

    float sum = 0.0f;

    if (idx < N) {
        sum += input[idx];
    }

    if (idx + blockDim.x < N) {
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

    if (laneId == 0) {
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

    if (warpId == 0) {

        const int numWarps =
            (blockDim.x + warpSize - 1)
            / warpSize;


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

        if (laneId == 0) {
            output[blockIdx.x] = blockSum;
        }
    }
}
```

## 是否是合并访存？
是，因为加载的地址都是连续的，所以是合并访存。
什么是合并访存？  
**内存事务的合并**  
- 当 warp 中所有线程访问的地址连续排列，并且起始地址对齐到事务大小的整数倍时，这些访问就是完全合并的。
- 例如，假设一个事务大小为 128 字节，一个 float 占 4 字节，那么 32 个线程连续读取 32 个 float（共 128 字节）就是一次合并访问，只需 1 次内存事务。

## 是否重复计算？
没有重复计算，  
也就是该线程负责的第一个元素仍有效，但第二个元素已越界。此时它只加第一个元素一次；前面的 block 和其他线程都不会处理这个 idx。
## Reduction 很适合用来理解 CUDA 的层级：

```
Thread
│
│ register sum
▼

Warp
│
│ __shfl_down_sync()
▼

Block
│
│ shared memory + __syncthreads()
▼

Grid
│
│ partial sums
▼

最终结果
```
这实际上是 CUDA 高性能算子的一个非常普遍的设计模式：
> 能在线程 Register 里处理的，不放 Shared；能在 Warp 里处理的，不做 Block 同步；只有跨 Warp 时才使用 Shared Memory + __syncthreads()。
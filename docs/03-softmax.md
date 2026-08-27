**下面给你一个适合学习的 CUDA Softmax 实现：按行做 softmax。**  
假设输入是一个二维矩阵：  
X: [num_rows, num_cols]  
对每一行做：  

$ softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x)) $  

这里减去 max(x) 是为了数值稳定，防止 exp() 溢出。  



## 这版 Softmax 的核心思想

一个 block 负责一行：

```text
Block 0 -> row 0
Block 1 -> row 1
Block 2 -> row 2
```

每个 block 内部的线程合作完成三件事：

```text
1. 求这一行的 max
2. 求 sum(exp(x - max))
3. 写回 exp(x - max) / sum
```

所以 kernel 的结构是：

```text
Global Memory
    ↓
每个 thread 处理若干列！！
    ↓
blockReduceMax
    ↓
blockReduceSum
    ↓
写回 softmax
```

---

## 为什么要减去 max？

直接算：

```cpp
expf(x)
```

可能溢出。

比如：

```cpp
expf(1000.0f)
```

会变成无穷大。

所以 softmax 通常写成：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

这个结果和原始 softmax 数学上等价，但数值更稳定。

---

## 这份代码和 reduction 的关系

Softmax 本质上就是两个 reduction：

```text
第一轮 reduction:
max(x)

第二轮 reduction:
sum(exp(x - max))
```

所以你前面学的 reduction 正好用上了。

这里的 `blockReduceMax` 和 `blockReduceSum` 都用了：

```cpp
__shfl_down_sync()
```

做 warp 内归约，然后用少量 shared memory 汇总多个 warp 的结果。

---

## 这版适合什么场景？

这个实现适合学习和中等长度行，比如：

```text
num_cols = 128, 256, 512, 1024, 2048, 4096
```

如果是大模型里的 attention softmax，通常还有更多优化，比如：

- 向量化 load/store
- half / bfloat16
- warp-level softmax
- block-level softmax
- online softmax
- 和 matmul 融合，也就是 FlashAttention 思想

但你现在第一版先把这个 **row-wise stable softmax** 写熟，是最合理的。
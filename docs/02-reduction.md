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
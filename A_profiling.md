Vector Add          ✅
        |
Reduction          ✅
        |
Softmax            ✅
        |
        ↓
LayerNorm           ⭐⭐⭐⭐⭐
        |
        ↓
Transpose           ⭐⭐⭐⭐
        |
        ↓
GEMM                ⭐⭐⭐⭐⭐
        |
        ↓
Fused Softmax       ⭐⭐⭐⭐⭐
        |
        ↓
FlashAttention 1/2/3      ⭐⭐⭐⭐⭐
        |
        ↓
Quantization Kernel
        |
        ↓
Inference Kernels

prefill：GEMM
decode：GEMV

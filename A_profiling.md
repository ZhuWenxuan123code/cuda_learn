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
FlashAttention      ⭐⭐⭐⭐⭐
        |
        ↓
Quantization Kernel
        |
        ↓
Inference Kernels

prefill：GEMM
decode：GEMV
# CUDA backend coverage

Legend: **optimized** uses cuBLAS or a focused kernel; **reference CUDA** is a
correctness/capacity baseline; **composed** builds a public operation from
smaller CUDA operations; **parity pending** means real released fixtures have
not yet been compared on NVIDIA hardware.

| Operation family | Implementation | Status |
|---|---|---|
| Device/tensor lifecycle | two streams, ready/last-use events, tracked CUDA allocation budget | functional; real-GPU stress pending |
| Low-VRAM weight storage | VRAM LRU → RAM LRU → safetensors fallback | host policy tested; CUDA execution pending |
| Generated INT8 retention | D2H authoritative RAM copy before VRAM eviction | source syntax/static coverage; real-GPU pending |
| BF16/FP32 linear | cuBLAS Tensor Core GEMM, FP32 accumulation | optimized; parity pending |
| Mixed patch projection | shared-memory tiled CUDA GEMM | reference CUDA; parity pending |
| Dynamic INT8 linear | row/weight scales, cuBLAS INT8→INT32, scale epilogue | optimized; parity pending |
| SiLU/GELU/GEGLU/SwiGLU | elementwise CUDA | optimized; parity pending |
| RMSNorm/LayerNorm | one block per row, FP32 reduction | optimized; parity pending |
| AdaLN/gating/fused boundaries | CUDA plus composed fallbacks | composed; parity pending |
| QKV split, head RMS and RoPE | head-major CUDA kernels | optimized; parity pending |
| SDPA/GQA/causal attention | one-pass bounded-memory online-softmax | reference CUDA; parity pending |
| Token pool/expand | mapping CUDA kernels | reference CUDA; parity pending |
| Embedding and Euler sampler | CUDA | optimized; parity pending |
| Conv1d/transposed Conv1d | generic channels-last CUDA | reference CUDA; parity pending |
| Conv3d | generic channels-last CUDA | reference CUDA; parity pending |
| VAE pad/group norm | CUDA | reference CUDA; parity pending |
| Snake1d | learned periodic CUDA activation | optimized; parity pending |
| Alias-free Snake | fused released 12-tap polyphase FIR + SnakeBeta | CPU identity tested; GPU parity pending |
| Portable tokenizer | ICU NFC/categories + yyjson BPE | host C; fixture parity pending |
| RGB24 resize | separable Lanczos3 | host C; unit tested |

## Memory safety invariants

1. An offloaded tensor is consumable only after its upload-ready event.
2. A resident weight cannot be evicted while pinned by the current operation.
3. Eviction waits for the tensor's recorded last-use event.
4. Generated INT8 tensors are not offloadable until an authoritative RAM copy
   exists.
5. RAM LRU eviction only drops tensors that retain a valid file source.
6. Scratch and ordinary tensor allocations share the same hard VRAM budget.

## API audit

The pinned `h3_gpu.h` contains 103 exported backend functions:

```bash
python3 scripts/verify_backend_api.py \
  --header third_party/h3/h3_gpu.h
```

Interface coverage does not prove numerical parity; follow `docs/VALIDATION.md`.

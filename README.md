# SpectralQ

**6-bit DCT quantization of LLM weights via a chunk-major CUDA kernel.**

Decomposes `nn.Linear` weights into blocks, transforms each block via the DCT-II, quantizes the frequency coefficients uniformly to 6 bits, and computes the forward pass *directly in the frequency domain* — no weight matrix materialization.

---

## How it works

Every weight `W[out_f, in_f]` is reshaped into blocks of `block_size` (default 256). Each block's 256 values are transformed by the orthonormal DCT-II basis. Uniform 6-bit quantization is applied per frequency bin:

```python
coeffs[block][k] = DCT(block)[k]
qcoeff[block][k] = round((coeffs - min_k) / scale_k), clamped to [0, 63]
```

The forward pass has two modes:

**CUDA kernel path** (primary): two kernels per token —
1. **Projection** — DCT of the input: `n_chunks` blocks project `x` chunks into frequency space.
2. **Chunk-major GEMV** — One block per output row reads chunked, packed coefficients (5 × 6-bit per `uint32`) and accumulates the dot product with the frequency-domain input.

**Python fallback path**: dequantizes coefficients to fp16 once (lazy, first forward call), merges DC and AC into a single fp16 tensor, then runs one fused `F.linear` per layer — no per-call dequantization overhead.

Coefficients are stored `[n_chunks, out_f, pk]` — all output rows read the same contiguous chunk slab, maximizing L2 reuse. Naive `[out_f, n_chunks, pk]` thrashes L2 to ~10% of peak bandwidth; chunk-major layout achieves **61%** on RTX 4060.

Supported block sizes: 64, 128, 256, 512.

---

## Benchmarks (BS=1, RTX 4060 8 GB)

| Model | Base FP16 | DCT 6-bit | Slowdown | VRAM saved |
|---|---|---|---|---|
| SmolLM-1.7B | 28.3 ms, 3.43 GB | 38.3 ms, 2.04 GB | **1.35×** | **−41%** |
| Qwen2.5-3B | 44.5 ms, 6.30 GB | 74.7 ms, 3.36 GB | **1.68×** | **−47%** |

For BS > 1, the kernel loops over tokens in Python (see Limitations).

---

## Quick start

```bash
# 1. Compile the CUDA kernel
cd SpectralQ/csrc
# Windows:
.\compile.ps1
# Linux:
./compile.sh

# 2. Install dependencies
pip install -r requirements.txt

# 3. Quantize and benchmark BS=1
python scripts/quantize.py --model HuggingFaceTB/SmolLM-1.7B

# 4. Generate text
python scripts/generate.py --model HuggingFaceTB/SmolLM-1.7B \
    --prompt "The future of AI is"
```

---

## Limitations

- **Token-by-token batch**: The kernel processes one input token per CUDA launch. BS > 1 loops in Python — fine for autoregressive generation (BS=1), slower for benchmarking large batches.
- **CUDAGraph incompatible**: ctypes kernel launches bypass the PyTorch dispatcher — CUDAGraph capture will not work.
- **Requires compilation**: The CUDA kernel must be compiled with nvcc + MSVC (Windows) or g++ (Linux) before use.
- **Pack before forward**: `pack_and_prep_model()` must be called before the first forward pass — the Python fallback path frees `qcoeff_uint8` after converting to fused fp16 coefficients.

---

## Citation

```bibtex
@misc{spectralq2025,
  title = {SpectralQ: 6-bit DCT Quantization via Chunk-Major CUDA Kernels},
  year = {2025}
}
```

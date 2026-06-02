"""Convert a HuggingFace bf16 model to 6-bit DCT and benchmark single-token latency."""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectralq import convert_to_full_rank, pack_and_prep_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-1.7B")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--coeff-bits", type=int, default=6)
    parser.add_argument("--benchmark", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(device)
    model.eval()

    if args.benchmark:
        dummy = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
        for _ in range(3):
            model(dummy, use_cache=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            model(dummy, use_cache=False)
        torch.cuda.synchronize()
        t_base = (time.perf_counter() - t0) / 20
        base_vram = torch.cuda.max_memory_allocated()
        print(f"Base FP16: {t_base*1000:.1f} ms  ({1/t_base:.1f} tok/s)  VRAM: {base_vram/1e9:.2f} GB")

    n = convert_to_full_rank(model, block_size=args.block_size,
                             coeff_bits=args.coeff_bits, quant_type="uniform")
    print(f"Converted {n} DCT layers", flush=True)
    pack_and_prep_model(model)

    torch.cuda.reset_peak_memory_stats()

    if args.benchmark:
        for _ in range(3):
            model(dummy, use_cache=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            model(dummy, use_cache=False)
        torch.cuda.synchronize()
        t_dct = (time.perf_counter() - t0) / 20
        dct_vram = torch.cuda.max_memory_allocated()
        ratio = t_dct / t_base
        vram_saved = (base_vram - dct_vram) / 1e9
        vram_pct = (1 - dct_vram / base_vram) * 100
        print(f"DCT 6-bit: {t_dct*1000:.1f} ms  ({1/t_dct:.1f} tok/s)  "
              f"VRAM: {dct_vram/1e9:.2f} GB")
        print(f"vs base: {ratio:.2f}x slowdown, VRAM -{vram_saved:.2f} GB ({vram_pct:.0f}%)")


if __name__ == "__main__":
    main()

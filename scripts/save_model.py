"""Quantize a model to 6-bit DCT and save as safetensors for HuggingFace upload."""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectralq import convert_to_full_rank, pack_and_prep_model, save_quantized


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-1.7B")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--coeff-bits", type=int, default=6)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        model_short = args.model.split("/")[-1].replace(".", "_")
        args.save_dir = f"dct_{model_short}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model._tokenizer = tokenizer
    model.to(device)
    model.eval()

    n = convert_to_full_rank(model, block_size=args.block_size,
                             coeff_bits=args.coeff_bits, quant_type="uniform")
    print(f"Converted {n} DCT layers", flush=True)

    if n == 0:
        print("ERROR: no layers were converted! Check model architecture.")
        return

    pack_and_prep_model(model)
    print(f"Packed. Packed buffer shape sample: {next((m._qcoeff_packed.shape for m in model.modules() if isinstance(m, type(list(model.modules())[0])) and hasattr(m, '_qcoeff_packed')), 'N/A')}", flush=True)

    save_quantized(model, args.save_dir)
    print(f"\nSaved to {args.save_dir}/")
    print(f"Upload the entire directory to HuggingFace to share the quantized model.")


if __name__ == "__main__":
    main()

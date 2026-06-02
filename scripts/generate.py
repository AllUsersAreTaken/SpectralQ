"""Generate text with a DCT-quantized model."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectralq import convert_to_full_rank, pack_and_prep_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-1.7B")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--coeff-bits", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    n = convert_to_full_rank(model, block_size=args.block_size,
                             coeff_bits=args.coeff_bits, quant_type="uniform")
    print(f"Converted {n} DCT layers", flush=True)
    pack_and_prep_model(model)

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n{text}")


if __name__ == "__main__":
    main()

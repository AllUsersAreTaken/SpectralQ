"""Generate text with a DCT-quantized model and optional spectral KV cache."""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectralq import convert_to_full_rank, pack_and_prep_model, compress_kv_cache


def _sample(logits, temperature, top_k=None):
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1:]] = -float("Inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-1.7B")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--coeff-bits", type=int, default=6)
    parser.add_argument("--spectral-kv", action="store_true",
                        help="Enable spectral KV cache compression")
    parser.add_argument("--kv-window", type=int, default=512,
                        help="KV cache window before compression triggers")
    parser.add_argument("--kv-keep-ratio", type=float, default=0.5,
                        help="Fraction of DCT coefficients to keep (0-1)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    n = convert_to_full_rank(model, block_size=args.block_size,
                             coeff_bits=args.coeff_bits, quant_type="uniform")
    print(f"Converted {n} DCT layers", flush=True)
    pack_and_prep_model(model)

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = args.prompt

    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.input_ids.shape[1]

    # --- spectral KV: custom loop ---
    if args.spectral_kv:
        def _clen(c):
            if hasattr(c, 'get_seq_length'):
                return c.get_seq_length()
            return c[0][0].shape[2]

        print(f"Spectral KV: window={args.kv_window}, "
              f"keep_ratio={args.kv_keep_ratio}", flush=True)

        past_key_values = None
        generated = input_ids.input_ids
        total_tokens = input_ids.input_ids.shape[1]
        kv_positions = torch.arange(total_tokens, device='cpu', dtype=torch.long)
        tok_this = 0
        t0 = time.perf_counter()

        # prefill
        out = model(generated, use_cache=True, past_key_values=past_key_values)
        past_key_values = out.past_key_values
        next_id = _sample(out.logits, args.temperature, args.top_k)
        generated = torch.cat([generated, next_id], dim=-1)
        total_tokens += 1
        tok_this = 1

        # decode loop
        for step in range(args.max_new_tokens - 1):
            pos = torch.tensor([[total_tokens]], device=device)
            out = model(generated[:, -1:], use_cache=True,
                        past_key_values=past_key_values,
                        position_ids=pos)
            past_key_values = out.past_key_values
            total_tokens += 1
            kv_positions = torch.cat([kv_positions, torch.tensor([total_tokens - 1])])
            next_id = _sample(out.logits, args.temperature, args.top_k)
            generated = torch.cat([generated, next_id], dim=-1)
            tok_this += 1

            if _clen(past_key_values) > args.kv_window:
                before = _clen(past_key_values)
                past_key_values, kv_positions = compress_kv_cache(
                    past_key_values, model=model, sink_tokens=4,
                    keep_ratio=args.kv_keep_ratio,
                    orig_position_ids=kv_positions)
                after = _clen(past_key_values)
                if after < before:
                    print(f"  step {total_tokens-1}: KV cache {before} -> {after} "
                          f"({(1-after/before)*100:.0f}% reduction)", flush=True)

        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n{text}")
        print(f"\n--- {tok_this} tokens in {elapsed:.2f}s "
              f"({tok_this/elapsed:.1f} tok/s) ---", flush=True)

    # --- standard path ---
    else:
        out = model.generate(
            **input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_k=args.top_k or 50,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n{text}")


if __name__ == "__main__":
    main()

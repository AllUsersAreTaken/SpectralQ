import json
import os
import torch
from transformers import AutoConfig
from .modules import FactShieldHarmonicLinear, convert_to_full_rank


def save_quantized(model, save_dir, shard_size_gb=2):
    """Save a DCT-quantized model as safetensors ready for HuggingFace upload.

    The saved checkpoint can be loaded back with ``from_pretrained_quantized``.
    It includes the original model weights for non-DCT layers plus the packed
    DCT coefficients, scale/zero, and DCT basis for each converted layer.
    """
    os.makedirs(save_dir, exist_ok=True)

    model.config.save_pretrained(save_dir)
    tokenizer = getattr(model, "_tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

    dct_config = {
        "spectralq_version": 1,
        "dct_layers": [],
    }
    for name, mod in model.named_modules():
        if isinstance(mod, FactShieldHarmonicLinear):
            dct_config["dct_layers"].append({
                "name": name,
                "out_features": mod.out_features,
                "in_features": mod.in_features,
                "block_size": int(mod.block_size),
                "coeff_bits": mod.coeff_bits,
                "padded_in_f": getattr(mod, "_padded_in_f", mod.in_features),
            })

    with open(os.path.join(save_dir, "spectralq_config.json"), "w") as f:
        json.dump(dct_config, f, indent=2)

    state = model.state_dict()

    state_to_save = {}
    for k, v in state.items():
        is_dct = any(layer["name"] in k for layer in dct_config["dct_layers"])
        if is_dct or not any(k.startswith(layer["name"]) for layer in dct_config["dct_layers"]):
            state_to_save[k] = v.contiguous().clone() if v.is_contiguous() else v.clone()

    if len(state_to_save) == 0:
        print("  WARNING: empty state_dict to save!")
    else:
        from safetensors.torch import save_file as st_save
        st_save(state_to_save, os.path.join(save_dir, "model.safetensors"))
        total_gb = sum(v.numel() * v.element_size() for v in state_to_save.values()) / 1e9
        print(f"  saved {len(state_to_save)} tensors ({total_gb:.2f} GB)")


def from_pretrained_quantized(model_id, checkpoint_dir, device="cuda", block_size=256):
    """Load a DCT-quantized model from safetensors, ready for inference.

    Args:
        model_id: HuggingFace model ID (e.g. ``HuggingFaceTB/SmolLM-1.7B``)
        checkpoint_dir: Directory containing ``model.safetensors`` and ``spectralq_config.json``
        device: Target device
        block_size: DCT block size used during quantization

    Returns:
        model with DCT layers populated from checkpoint; call ``pack_and_prep_model``
        before the first forward pass.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()

    print("Converting to DCT...")
    convert_to_full_rank(model, block_size=block_size, coeff_bits=6, quant_type="uniform")

    from safetensors.torch import load_file as st_load
    print(f"Loading DCT weights from {checkpoint_dir}...")
    state = st_load(os.path.join(checkpoint_dir, "model.safetensors"))

    _qcoeff_keys = [k for k in state if k.endswith("_qcoeff_packed")]

    for k in _qcoeff_keys:
        mod = model.get_submodule(k.replace("._qcoeff_packed", ""))
        t = state.pop(k)
        mod_dev = next(mod.buffers()).device
        mod._buffers["_qcoeff_packed"] = t.to(device=mod_dev)

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_dct = sum(1 for m in model.modules() if isinstance(m, FactShieldHarmonicLinear))
    print(f"  DCT modules: {n_dct}, loaded {len(_qcoeff_keys)} packed-coeff tensors")

    return model

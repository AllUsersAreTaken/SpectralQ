import math
import torch

_dct_matrices = {}


def _dct_matrix(N, device):
    key = (N, str(device))
    if key in _dct_matrices:
        return _dct_matrices[key]
    basis = torch.empty(N, N, device=device, dtype=torch.float32)
    i = torch.arange(N, device=device, dtype=torch.float32)
    for k in range(N):
        if k == 0:
            basis[k] = 1.0 / math.sqrt(N)
        else:
            basis[k] = math.sqrt(2.0 / N) * torch.cos(
                math.pi * k * (2.0 * i + 1.0) / (2.0 * N))
    _dct_matrices[key] = basis
    return basis


def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_transform(x, cos, sin, unrotate=False):
    if unrotate:
        return x * cos - _rotate_half(x) * sin
    return x * cos + _rotate_half(x) * sin


def _extract_pairs(past_key_values):
    from transformers.cache_utils import DynamicCache
    if isinstance(past_key_values, DynamicCache):
        return ([l.keys for l in past_key_values.layers],
                [l.values for l in past_key_values.layers],
                True)
    return [k for k, v in past_key_values], [v for k, v in past_key_values], False


def _build_cache(keys, values, is_dynamic):
    if is_dynamic:
        from transformers.cache_utils import DynamicCache, DynamicLayer
        dc = DynamicCache()
        dc.layers.clear()
        for k, v in zip(keys, values):
            dc.layers.append(DynamicLayer.from_tensors(k, v))
        return dc
    return tuple(zip(keys, values))


def compress_kv_cache(past_key_values, model=None, sink_tokens=4, keep_ratio=0.5,
                      orig_position_ids=None):
    """Compress the non-sink region of the KV cache via DCT low-pass.

    If ``model`` and ``orig_position_ids`` are both provided, keys are
    unrotated to content space before DCT and re-rotated with interpolated
    position IDs afterward.  This prevents RoPE-position drift from
    accumulating across multiple compression cycles.

    Args:
        past_key_values: ``DynamicCache`` or tuple-of-tuples.
        model: the HF model (used to access ``rotary_emb``).  If ``None``,
               keys are compressed in post-rotation space (legacy behaviour).
        sink_tokens: number of initial tokens left untouched.
        keep_ratio: fraction of DCT coefficients to keep.
        orig_position_ids: 1-D long tensor with the position of each token
            in the current cache (length = cache length).  Ignored when
            ``model is None``.

    Returns:
        ``(new_cache, new_position_ids)`` where ``new_position_ids`` is a
        1-D long tensor matching the compressed cache length.
    """
    rotary_emb = None
    has_rope = False
    if model is not None and orig_position_ids is not None:
        # recent transformers puts rotary_emb on model.model.rotary_emb
        rope_src = (getattr(getattr(model, 'model', None), 'rotary_emb', None) or
                    getattr(model, 'rotary_emb', None))
        if rope_src is not None:
            rotary_emb = rope_src
            has_rope = True

    keys, values, is_dynamic = _extract_pairs(past_key_values)
    B, H, _, D = keys[0].shape
    dt = keys[0].dtype
    dev = keys[0].device

    # Find the majority sequence length (all layers should be close)
    T_counts = {}
    for k in keys:
        T_counts[k.shape[2]] = T_counts.get(k.shape[2], 0) + 1
    T = max(T_counts, key=T_counts.get)
    non_sink_len = T - sink_tokens
    keep_len = max(2, int(non_sink_len * keep_ratio))

    if non_sink_len <= 2 or keep_len >= non_sink_len:
        out = (orig_position_ids.clone() if orig_position_ids is not None
               else torch.arange(T))
        return past_key_values, out

    # --- new position IDs (interpolated) ---
    if orig_position_ids is not None:
        orig_end = orig_position_ids[-1].item()
    else:
        orig_end = T - 1

    compressed_pos = torch.linspace(
        sink_tokens, orig_end, keep_len,
        device='cpu', dtype=torch.float).round().long().clamp_(sink_tokens, orig_end)
    compressed_pos = compressed_pos.unique(sorted=True)
    if len(compressed_pos) < keep_len:
        extra = keep_len - len(compressed_pos)
        pad = torch.arange(orig_end - extra + 1, orig_end + 1,
                           device='cpu', dtype=torch.long)
        compressed_pos = torch.cat([compressed_pos, pad])
    new_position_ids = torch.cat([
        torch.arange(sink_tokens, device='cpu', dtype=torch.long),
        compressed_pos[:keep_len],
    ])

    new_keys, new_values = [], []

    for k, v in zip(keys, values):
        Tk = k.shape[2]
        if Tk <= sink_tokens + 2:
            new_keys.append(k); new_values.append(v)
            continue

        nsl = Tk - sink_tokens
        kl = max(2, int(nsl * keep_ratio))
        if kl >= nsl:
            new_keys.append(k); new_values.append(v)
            continue

        k_ns = k[:, :, sink_tokens:, :].contiguous()
        v_ns = v[:, :, sink_tokens:, :].contiguous()

        if has_rope:
            op_ns = orig_position_ids[sink_tokens:sink_tokens + nsl].to(dev).unsqueeze(0)
            dummy = torch.empty(1, 1, nsl, D, device=dev, dtype=dt)
            cu, su = rotary_emb(dummy, op_ns)
            k_ns = _rope_transform(k_ns, cu.unsqueeze(1), su.unsqueeze(1), unrotate=True)

        dct_f = _dct_matrix(nsl, dev)
        dct_l = _dct_matrix(kl, dev)
        k_freq = torch.einsum('kt,bhtd->bhkd', dct_f, k_ns.float())
        v_freq = torch.einsum('kt,bhtd->bhkd', dct_f, v_ns.float())
        k_rc = torch.einsum('tk,bhkd->bhtd', dct_l.T, k_freq[:, :, :kl]).to(dt)
        v_rc = torch.einsum('tk,bhkd->bhtd', dct_l.T, v_freq[:, :, :kl]).to(dt)

        if has_rope:
            np_ns = new_position_ids[sink_tokens:sink_tokens + kl].to(dev).unsqueeze(0)
            dummy2 = torch.empty(1, 1, kl, D, device=dev, dtype=dt)
            cr, sr = rotary_emb(dummy2, np_ns)
            k_rc = _rope_transform(k_rc, cr.unsqueeze(1), sr.unsqueeze(1), unrotate=False)

        new_keys.append(torch.cat([k[:, :, :sink_tokens], k_rc], dim=2))
        new_values.append(torch.cat([v[:, :, :sink_tokens], v_rc], dim=2))

    return _build_cache(new_keys, new_values, is_dynamic), new_position_ids


def rebuild_position_ids(past_key_values, sink_tokens=4):
    L = past_key_values.get_seq_length()
    return torch.arange(L, device='cpu', dtype=torch.long)

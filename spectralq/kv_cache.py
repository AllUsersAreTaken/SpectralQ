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


def _extract_pairs(past_key_values):
    """Return (keys_list, values_list, is_dynamic) from any cache format."""
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


def compress_kv_cache(past_key_values, sink_tokens=4, keep_ratio=0.5,
                      orig_seq_len=None):
    """Compress the non-sink region of the KV cache via DCT low-pass.

    For each layer::
        k[B,H,T,D]  -> DCT along T -> keep first L coeffs -> L-point IDCT
                     -> k[B,H,L,D]  (concatenated after sink tokens)

    Args:
        past_key_values: ``DynamicCache`` or tuple-of-tuples.
        sink_tokens: number of initial tokens left untouched.
        keep_ratio: fraction of DCT coefficients to keep.
        orig_seq_len: total tokens generated so far (used to compute correct
                      position IDs for compressed tokens).  If ``None``,
                      defaults to the current cache length (no position fix).

    Returns:
        ``(new_cache, new_position_ids)`` where ``new_position_ids`` is a
        1-D long tensor of length ``sink_tokens + keep_len`` with the
        position ID for each slot in the compressed cache.
    """
    keys, values, is_dynamic = _extract_pairs(past_key_values)
    new_keys, new_values = [], []
    first_T = None

    for k, v in zip(keys, values):
        B, H, T, D = k.shape
        if first_T is None:
            first_T = T
        if T <= sink_tokens + 2:
            new_keys.append(k)
            new_values.append(v)
            continue

        non_sink_len = T - sink_tokens
        keep_len = max(2, int(non_sink_len * keep_ratio))
        if keep_len >= non_sink_len:
            new_keys.append(k)
            new_values.append(v)
            continue

        k_ns = k[:, :, sink_tokens:, :].contiguous()
        v_ns = v[:, :, sink_tokens:, :].contiguous()

        dct_full = _dct_matrix(non_sink_len, k.device)
        dct_L = _dct_matrix(keep_len, k.device)
        dt = k_ns.dtype

        k_freq = torch.einsum('kt,bhtd->bhkd',
                              dct_full, k_ns.float())
        v_freq = torch.einsum('kt,bhtd->bhkd',
                              dct_full, v_ns.float())
        k_recon = torch.einsum('tk,bhkd->bhtd',
                               dct_L.T, k_freq[:, :, :keep_len]).to(dt)
        v_recon = torch.einsum('tk,bhkd->bhtd',
                               dct_L.T, v_freq[:, :, :keep_len]).to(dt)

        k_new = torch.cat([k[:, :, :sink_tokens], k_recon], dim=2)
        v_new = torch.cat([v[:, :, :sink_tokens], v_recon], dim=2)
        new_keys.append(k_new)
        new_values.append(v_new)

    new_cache = _build_cache(new_keys, new_values, is_dynamic)

    # --- position IDs ---
    cache_len = new_cache.get_seq_length()
    total_len = orig_seq_len if orig_seq_len is not None else first_T
    sink = min(sink_tokens, cache_len)
    compressed_len = cache_len - sink

    if compressed_len > 0 and total_len > sink:
        # Interpolate compressed positions across the original range
        orig_end = total_len - 1
        compressed_pos = torch.linspace(
            sink, orig_end, compressed_len,
            device='cpu', dtype=torch.long).clamp_(sink, orig_end)
        if compressed_len > 1:
            compressed_pos = compressed_pos.round().long()
        else:
            compressed_pos = compressed_pos.long()
        # Ensure unique sorted
        compressed_pos = compressed_pos.unique(sorted=True)
        # If dedup reduced size, pad with linearly spaced from end
        if len(compressed_pos) < compressed_len:
            extra = compressed_len - len(compressed_pos)
            pad = torch.arange(orig_end - extra + 1, orig_end + 1,
                               device='cpu', dtype=torch.long)
            compressed_pos = torch.cat([compressed_pos, pad])
        position_ids = torch.cat([
            torch.arange(sink, device='cpu', dtype=torch.long),
            compressed_pos[:compressed_len],
        ])
    else:
        position_ids = torch.arange(cache_len, device='cpu', dtype=torch.long)

    return new_cache, position_ids


def rebuild_position_ids(past_key_values, sink_tokens=4):
    """Return position IDs for the current cache (sequential fallback)."""
    L = past_key_values.get_seq_length()
    return torch.arange(L, device='cpu', dtype=torch.long)

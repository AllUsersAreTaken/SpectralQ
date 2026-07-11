"""DCT-guided token selection: keep exact tokens, drop low-importance ones."""
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


def dct_token_importance(keys, dct_matrix, top_m_ratio=0.25):
    """Score each token by its contribution to the most energetic DCT coefficients.

    importance[t] = ||key[t]||^2 * sum_{k in top_m} DCT(k, t)^2

    Args:
        keys: list of [B, H, T, D] tensors (one per layer).
        dct_matrix: [T, T] DCT matrix.
        top_m_ratio: fraction of top DCT coefficients to consider.

    Returns:
        importance: [B, T] per-token importance scores (averaged across heads and layers).
    """
    B, H, T, D = keys[0].shape
    m = max(1, int(T * top_m_ratio))

    # Average importance across reference layers
    all_importance = []
    for ki in keys:
        k_freq = torch.einsum('kt,bhtd->bhkd', dct_matrix, ki.float())  # [B, H, T, D]
        mag = k_freq.abs().mean(dim=-1)  # [B, H, T]
        _, top_idx = torch.topk(mag, m, dim=-1)  # [B, H, m]

        # basis_energy[t] = sum_{k in top_m} DCT(k, t)^2
        B_act, H_act = ki.shape[0], ki.shape[1]
        dct_e = dct_matrix.unsqueeze(0).unsqueeze(0).expand(B_act, H_act, -1, -1)
        idx_e = top_idx.unsqueeze(-1).expand(-1, -1, -1, T)
        selected = torch.gather(dct_e, dim=2, index=idx_e)  # [B, H, m, T]
        basis_energy = selected.pow(2).sum(dim=2)  # [B, H, T]

        key_norm_sq = ki.float().norm(dim=-1).pow(2)  # [B, H, T]
        imp = key_norm_sq * basis_energy  # [B, H, T]
        all_importance.append(imp)

    # Average across reference layers and heads, then across batch
    stacked = torch.stack(all_importance)  # [n_layers, B, H, T]
    return stacked.mean(dim=(0, 2))  # [B, T]


def compress_kv_cache_select(past_key_values, keep_ratio=0.5, sink_tokens=4,
                              top_m_ratio=0.25, reference_layers=None):
    """Drop low-importance tokens (DCT-guided), keep exact keys/values.

    Unlike :func:`compress_kv_cache`, this does NOT reconstruct via IDCT.
    It simply evicts the least important non-sink tokens, preserving their
    original positions.

    Args:
        past_key_values: ``DynamicCache`` or tuple-of-tuples.
        keep_ratio: fraction of non-sink tokens to retain.
        sink_tokens: number of initial tokens always kept.
        top_m_ratio: fraction of top DCT coefficients to use for importance.
        reference_layers: list of layer indices to use for importance computation.
            ``None`` uses all layers.

    Returns:
        ``(new_cache, new_position_ids)`` where ``new_position_ids`` is a
        1-D long tensor matching the selected cache length.
    """
    keys, values, is_dynamic = _extract_pairs(past_key_values)
    B, H, _, D = keys[0].shape
    dev = keys[0].device
    dt = keys[0].dtype

    # Find majority length
    T_counts = {}
    for k in keys:
        T_counts[k.shape[2]] = T_counts.get(k.shape[2], 0) + 1
    T = max(T_counts, key=T_counts.get)
    non_sink_len = T - sink_tokens
    keep_len = max(2, int(non_sink_len * keep_ratio))

    if non_sink_len <= 2 or keep_len >= non_sink_len:
        return past_key_values, torch.arange(T, device='cpu')

    # Importance from reference layers
    ref_layers = reference_layers or list(range(len(keys)))
    ref_keys = [keys[i] for i in ref_layers if i < len(keys)]

    dct_f = _dct_matrix(non_sink_len, dev)
    non_sink_keys = [k[:, :, sink_tokens:, :].contiguous() for k in ref_keys]
    importance = dct_token_importance(non_sink_keys, dct_f, top_m_ratio)  # [B, non_sink_len]

    # Select top-k non-sink positions
    B_act = importance.shape[0]
    new_keys, new_values = [], []
    new_positions = torch.arange(sink_tokens, device='cpu', dtype=torch.long)

    for b in range(B_act):
        imp = importance[b]  # [non_sink_len]
        _, top_idx = torch.topk(imp, keep_len, dim=-1)  # [keep_len]
        top_idx = top_idx.sort().values  # keep original order
        top_positions = (top_idx + sink_tokens).cpu()
        new_positions = torch.cat([new_positions, top_positions])

    new_positions = new_positions.unique(sorted=True)

    for k, v in zip(keys, values):
        Tk = k.shape[2]
        if Tk <= sink_tokens + 2:
            new_keys.append(k); new_values.append(v)
            continue

        # Select sink + kept positions
        sink_k = k[:, :, :sink_tokens, :]
        sink_v = v[:, :, :sink_tokens, :]

        # Non-sink positions to keep
        keep_t = (new_positions[new_positions >= sink_tokens] - sink_tokens).to(dev).long()
        if len(keep_t) == 0:
            new_keys.append(sink_k); new_values.append(sink_v)
            continue

        keep_k = k[:, :, sink_tokens:, :][:, :, keep_t, :]
        keep_v = v[:, :, sink_tokens:, :][:, :, keep_t, :]

        new_keys.append(torch.cat([sink_k, keep_k], dim=2))
        new_values.append(torch.cat([sink_v, keep_v], dim=2))

    return _build_cache(new_keys, new_values, is_dynamic), new_positions

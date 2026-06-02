import torch


def pack_module_coeffs(mod):
    """Pack and transpose a FactShieldHarmonicLinear module's 6-bit uint8
    coefficients into chunk-major int32 layout, then free the original.

    The kernel reads coefficients in the layout [n_chunks, out_f, pk],
    where pk = ceil(block_size / 5). Each int32 packs 5 × 6-bit values:

        bit 0-5:   slot 0   |  bit 6-11:  slot 1
        bit 12-17: slot 2   |  bit 18-23: slot 3
        bit 24-29: slot 4   |  bit 30-31: unused

    This layout ensures all SMs read the same contiguous chunk slab
    simultaneously, broadcasting through L2 instead of scattering across
    DRAM pages.
    """
    if hasattr(mod, '_qcoeff_packed') and mod._qcoeff_packed.numel() > 0:
        return
    u8 = mod.qcoeff_uint8
    out_f = mod.out_features
    bs = int(mod.block_size)
    in_f = getattr(mod, '_padded_in_f', mod.in_features)
    n_chunks = in_f // bs
    pk = (bs + 4) // 5

    u8 = u8.reshape(out_f, n_chunks, bs)

    pad = pk * 5 - bs
    if pad > 0:
        u8 = torch.nn.functional.pad(u8, (0, pad))

    as_5 = u8.view(out_f, n_chunks, pk, 5).to(torch.int32)
    packed = (as_5[:, :, :, 0] << 0) | \
             (as_5[:, :, :, 1] << 6) | \
             (as_5[:, :, :, 2] << 12) | \
             (as_5[:, :, :, 3] << 18) | \
             (as_5[:, :, :, 4] << 24)

    chunk_major = packed.permute(1, 0, 2).contiguous()
    mod._buffers['_qcoeff_packed'] = chunk_major
    mod._n_chunks = n_chunks
    mod._pk_per_chunk = pk
    mod._cm_out_features = out_f
    mod._cm_in_features = in_f
    mod._cm_block_size = bs

    mod._buffers.pop("qcoeff_uint8", None)
    try:
        del mod.qcoeff_uint8
    except AttributeError:
        pass

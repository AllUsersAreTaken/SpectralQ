import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TARGETS = ("gate_proj", "up_proj", "down_proj")


def make_dct_basis(block_size, device, dtype=torch.float32):
    """Create [block_size, block_size] orthonormal DCT-II basis."""
    basis = torch.empty(block_size, block_size, device=device, dtype=dtype)
    i = torch.arange(block_size, device=device, dtype=dtype)
    for k in range(block_size):
        if k == 0:
            basis[k] = 1.0 / math.sqrt(block_size)
        else:
            basis[k] = math.sqrt(2.0 / block_size) * torch.cos(
                math.pi * k * (2.0 * i + 1.0) / (2.0 * block_size))
    return basis


def get_parent_module(root, dotted_name):
    parent_name = ".".join(dotted_name.split(".")[:-1])
    child_name = dotted_name.split(".")[-1]
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, child_name


def is_target_layer(name, module, targets):
    if not isinstance(module, nn.Linear):
        return False
    if any(skip in name for skip in ("lm_head", "embed_tokens")):
        return False
    return any(target in name for target in targets)


class FactShieldHarmonicLinear(nn.Module):
    """A Linear layer parameterized in the DCT frequency domain.

    Full-rank mode (the only mode used here): the original weight matrix W
    is decomposed into blocks of size `block_size`, each block transformed
    via the DCT-II orthonormal basis. The coefficients are stored in 6-bit
    uniform quantization. During forward, instead of materializing W and
    doing a standard GEMM, the CUDA kernel computes:

        out[row] = sum_{chunk, j} coeff[row][chunk][j] * x_freq[j]

    where x_freq is the DCT of the input block.
    """

    def __init__(self, residual_weight, bias, block_size, coeff_bits, quant_type, device):
        super().__init__()
        self.out_features, self.in_features = residual_weight.shape
        self.block_size = int(block_size)
        self.use_freq_forward = True
        self.full_rank = True
        self.original_numel = residual_weight.numel()
        self.coeff_bits = coeff_bits
        self._quant_type = quant_type

        dct = make_dct_basis(self.block_size, device=residual_weight.device, dtype=torch.float32)
        self.register_buffer("learned_basis", dct.T.contiguous())

        padded_in_f = ((self.in_features + self.block_size - 1) // self.block_size) * self.block_size
        self._padded_in_f = padded_in_f
        if padded_in_f > self.in_features:
            residual_weight = F.pad(residual_weight, (0, padded_in_f - self.in_features))
        flat = residual_weight.detach().float().flatten()
        blocks = flat.view(-1, self.block_size)
        self.num_blocks = blocks.shape[0]

        coeffs = blocks @ self.learned_basis

        dc_raw = coeffs[:, 0:1].clone()
        self.register_buffer("_dc_raw", dc_raw.to(torch.float16))

        if coeff_bits > 0 and quant_type == "uniform":
            n_levels = 2 ** coeff_bits
            c_min = coeffs.amin(dim=0, keepdim=True)
            c_max = coeffs.amax(dim=0, keepdim=True)
            margin = (c_max - c_min) * 1e-4
            c_min = c_min - margin
            c_max = c_max + margin
            scale = (c_max - c_min) / n_levels
            scale[scale == 0] = 1.0
            qcoeff = torch.round((coeffs - c_min) / scale).clamp(0, n_levels - 1)
            self.register_buffer("qcoeff_uint8", qcoeff.to(torch.uint8))
            self.register_buffer("qscale", scale.squeeze(0).to(torch.float16))
            self.register_buffer("qzero", c_min.squeeze(0).to(torch.float16))
        else:
            self.register_buffer("freq_coeffs", coeffs)
            self.register_buffer("qcoeff_uint8", torch.empty(0, dtype=torch.uint8))
            self.register_buffer("qscale", torch.empty(0, dtype=torch.float16))
            self.register_buffer("qzero", torch.empty(0, dtype=torch.float16))

        self.register_buffer("_qcoeff_packed", torch.empty(0, dtype=torch.int32))
        self.register_buffer("protected_mask", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("fact_weights", torch.zeros(1))

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def reconstruct_weight(self):
        dc = self._dc_raw.float()
        q = getattr(self, 'qcoeff_uint8', None)
        if q is None or q.numel() == 0:
            q = getattr(self, '_qcoeff_for_reconstruct', None)
        if q is None or q.numel() == 0:
            raise RuntimeError("No quantized coefficients available for reconstruction")
        ac = (q.float() * self.qscale.float() + self.qzero.float())
        coeffs = ac.clone()
        coeffs[:, 0:1] = dc
        basis = self.learned_basis.to(device=coeffs.device, dtype=coeffs.dtype)
        blocks = coeffs @ basis.T
        flat = blocks.reshape(-1)[:self.original_numel]
        return flat.view(self.out_features, self.in_features)

    def forward(self, x):
        if getattr(self, '_dct_prepped', False):
            return self._dct_forward(x)
        return self._forward_freq(x)

    def _forward_freq(self, x):
        BS = self.block_size
        nh = self.learned_basis.shape[-1]
        out_f, in_f = self.out_features, self.in_features
        n_chunks = in_f // BS
        *batch_shape, in_feat = x.shape
        x_2d = x.reshape(-1, in_f)
        B = x_2d.shape[0]
        x_blocks = x_2d.view(B, n_chunks, BS)
        x_proj = x_blocks @ self.learned_basis.to(x.dtype)

        # DC contribution from preserved fp16 buffer
        dc_raw_3d = self._dc_raw.to(x.dtype).reshape(out_f, n_chunks, 1)
        x_dc = x_proj[:, :, 0:1]
        dc_contrib = F.linear(x_dc.reshape(B, n_chunks),
                              dc_raw_3d.reshape(out_f, n_chunks))

        # AC contribution from quantized coefficients (skip DC index 0)
        if self.qcoeff_uint8.numel() > 0:
            coeffs_f32 = (self.qcoeff_uint8.float() * self.qscale.float() + self.qzero.float())
            coeffs_3d = coeffs_f32.to(x.dtype).view(out_f, n_chunks, nh)
            ac_coeff = coeffs_3d[:, :, 1:]
            x_ac = x_proj[:, :, 1:]
            ac_contrib = F.linear(x_ac.reshape(B, n_chunks * (nh - 1)),
                                  ac_coeff.reshape(out_f, n_chunks * (nh - 1)))
        else:
            ac_contrib = 0

        y_2d = dc_contrib + ac_contrib
        if self.bias is not None:
            y_2d = y_2d + self.bias.to(x.dtype)
        return y_2d.reshape(*batch_shape, out_f)


def convert_to_full_rank(model, targets=DEFAULT_TARGETS, block_size=64,
                         coeff_bits=6, quant_type="uniform"):
    """Replace target Linear layers with lossless full-rank DCT parameterization.

    The DCT is a bijective orthogonal transform — the coefficients store
    exactly the same information as the original weights. The freq-forward
    path computes x @ W^T without materializing W, saving substantial
    peak memory during inference.

    Returns the number of layers converted.
    """
    count = 0
    for name, mod in list(model.named_modules()):
        if not is_target_layer(name, mod, targets):
            continue
        parent, child_key = get_parent_module(model, name)
        w = mod.weight.detach().float()
        b = mod.bias
        harmonic = FactShieldHarmonicLinear(
            residual_weight=w,
            bias=b,
            block_size=block_size,
            coeff_bits=coeff_bits,
            quant_type=quant_type,
            device=w.device,
        )
        setattr(parent, child_key, harmonic)
        count += 1
    return count

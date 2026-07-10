import os
import ctypes
import torch

from .packer import pack_module_coeffs
from .modules import FactShieldHarmonicLinear


def _load_cm_dll(dll_dir):
    dll = os.path.join(dll_dir, "dct_kernel_chunk_major.dll")
    if not os.path.exists(dll):
        return None
    k = ctypes.CDLL(dll)
    k.launch_dct_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    k.launch_dct_forward.restype = None
    return k


def _load_old_dll(dll_dir):
    dll = os.path.join(dll_dir, "dct_kernel.dll")
    if not os.path.exists(dll):
        return None
    k = ctypes.CDLL(dll)
    k.launch_dct_6bit_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    k.launch_dct_6bit_forward.restype = None
    return k


class _KernelCache:
    def __init__(self):
        self.cm_kernel = None
        self.old_kernel = None

    def get(self, dll_dir=None):
        if dll_dir is None:
            dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc")
        if self.cm_kernel is None and self.old_kernel is None:
            self.cm_kernel = _load_cm_dll(dll_dir)
            if self.cm_kernel is not None:
                return self.cm_kernel, True
            self.old_kernel = _load_old_dll(dll_dir)
            if self.old_kernel is not None:
                return self.old_kernel, False
            raise RuntimeError("No DCT kernel DLL found — compile csrc/dct_chunk_major_kernel.cu first")
        if self.cm_kernel is not None:
            return self.cm_kernel, True
        return self.old_kernel, False

    def reset(self):
        self.cm_kernel = None
        self.old_kernel = None


_kernels = _KernelCache()


def _prep_lin(lin, dll_dir):
    if hasattr(lin, '_dct_prepped'):
        return
    if not hasattr(lin, 'qcoeff_uint8') or lin.qcoeff_uint8.numel() == 0:
        raise RuntimeError(
            f"qcoeff_uint8 is missing on layer {id(lin):#x} — it was probably freed by "
            f"_forward_freq. Run pack_and_prep_model BEFORE the first forward pass.")
    dev = lin.qcoeff_uint8.device
    k, is_cm = _kernels.get(dll_dir)
    if is_cm:
        pack_module_coeffs(lin)
    else:
        lin._qh_u8 = lin.qcoeff_uint8.contiguous().reshape(-1)
    lin._basis_h = lin.learned_basis.contiguous().half().reshape(-1)
    lin._qscale_h = lin.qscale.contiguous()
    lin._qzero_h = lin.qzero.contiguous()
    lin._out_buf = torch.zeros(lin.out_features, device=dev, dtype=torch.float32)
    bs = int(lin.block_size)
    padded_in_f = getattr(lin, '_padded_in_f', lin.in_features)
    lin._cm_n_chunks = padded_in_f // bs
    lin._cm_padded_in_f = padded_in_f
    lin._cm_block_size = bs
    lin._dct_prepped = True


def create_dct_workspace(model):
    dev = next(model.parameters()).device
    max_in = 1
    for mod in model.modules():
        if isinstance(mod, FactShieldHarmonicLinear) and getattr(mod, 'coeff_bits', 0) > 0:
            padded = getattr(mod, '_padded_in_f', mod.in_features)
            max_in = max(max_in, padded)
    ws = torch.empty(max_in * 2, device=dev, dtype=torch.float16)
    setattr(model, '_dct_ws_x', ws[:max_in])
    setattr(model, '_dct_ws_freq', ws[max_in:])
    return ws


def _launch(lin, ws_x, ws_freq, stream_ptr, dll_dir):
    k, is_cm = _kernels.get(dll_dir)
    if is_cm:
        k.launch_dct_forward(
            ctypes.c_void_p(ws_x.data_ptr()),
            ctypes.c_void_p(lin._qcoeff_packed.data_ptr()),
            ctypes.c_void_p(lin._qscale_h.data_ptr()),
            ctypes.c_void_p(lin._qzero_h.data_ptr()),
            ctypes.c_void_p(lin._basis_h.data_ptr()),
            ctypes.c_void_p(ws_freq.data_ptr()),
            ctypes.c_void_p(lin._out_buf.data_ptr()),
            lin._cm_out_features, lin._cm_in_features,
            lin._cm_block_size, lin._pk_per_chunk,
            stream_ptr,
        )
    else:
        k.launch_dct_6bit_forward(
            ctypes.c_void_p(ws_x.data_ptr()),
            ctypes.c_void_p(lin._qh_u8.data_ptr()),
            ctypes.c_void_p(lin._qscale_h.data_ptr()),
            ctypes.c_void_p(lin._qzero_h.data_ptr()),
            ctypes.c_void_p(lin._basis_h.data_ptr()),
            ctypes.c_void_p(ws_freq.data_ptr()),
            ctypes.c_void_p(lin._out_buf.data_ptr()),
            lin.out_features, lin.in_features, int(lin.block_size),
            stream_ptr,
        )
    return lin._out_buf


def make_dct_forward(lin, model, dll_dir=None):
    _prep_lin(lin, dll_dir)
    ws_x = model._dct_ws_x
    ws_freq = model._dct_ws_freq
    r = lin
    stream = torch.cuda.current_stream()
    padded_in_f = r._cm_padded_in_f
    n_chunks = r._cm_n_chunks

    def f(self, x):
        *batch_shape, in_f = x.shape
        B = 1
        for s in batch_shape:
            B *= s
        if B == 0:
            return x
        stream_ptr = ctypes.c_void_p(stream.cuda_stream)
        if in_f < padded_in_f:
            x = torch.nn.functional.pad(x, (0, padded_in_f - in_f))
        if B == 1:
            ws_x[:padded_in_f].copy_(x.reshape(-1).float())
            r._out_buf.zero_()
            _launch(r, ws_x, ws_freq, stream_ptr, dll_dir)
            if len(batch_shape) == 0:
                return r._out_buf.to(x.dtype)
            return r._out_buf.unsqueeze(0).to(x.dtype)
        x_2d = x.reshape(-1, padded_in_f)
        out = torch.zeros(B, r.out_features, device=x.device, dtype=x.dtype)
        for i in range(B):
            ws_x[:padded_in_f].copy_(x_2d[i].float())
            r._out_buf.zero_()
            sp = ctypes.c_void_p(stream.cuda_stream)
            _launch(r, ws_x, ws_freq, sp, dll_dir)
            out[i] = r._out_buf.to(x.dtype)
        return out.reshape(*batch_shape, r.out_features).to(x.dtype)
    return f


def pack_and_prep_model(model, dll_dir=None):
    create_dct_workspace(model)
    for mod in model.modules():
        if isinstance(mod, FactShieldHarmonicLinear) and getattr(mod, 'coeff_bits', 0) > 0:
            mod.forward = make_dct_forward(mod, model, dll_dir).__get__(mod, type(mod))
            mod._free_python()

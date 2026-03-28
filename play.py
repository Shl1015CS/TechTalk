import math
from typing import Dict, Tuple
import torch
from task import input_t, output_t
from utils import make_match_reference

if not torch.cuda.is_available():
    raise RuntimeError("MI355x/Rocm GPU is required")

DEVICE = "cuda"
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}")

_IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
if _IS_ROCM:
    print(f"Rocm version: {torch.version.hip}")
else:
    print(f"CUDA version: {torch.version.cuda}")

MI350_NUM_XCDS = 8
MI350_CUS_PER_XCD = 32
MI350_TOTAL_CUS = MI350_NUM_XCDS * MI350_CUS_PER_XCD
MI350_L2_PER_XCD_KB = 4096
MI350_INF_CACHE_MB = 128

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
MXFP4_BLOCK = 32
MXFP4_E2M1_MAX = 6.0
NUM_KV_SPLITS = 32

def _pick_fp8_dtype() -> torch.dtype:
    # For simplicity, we use the same FP8 format for all tensors.
    # In practice, you might want to choose different formats based on the tensor distribution.
    if _IS_ROCM and hasattr(torch, "float8_e4m3fnuz"):
        dt = torch.float8_e4m3fnuz
        try:
            # Test if the dtype is supported on this ROCm version
            _ = torch.tensor([0.0], device=DEVICE, dtype =dt)
            return dt
        except Exception as e:
            pass
    if hasattr(torch, "float8_e4m3fn"):
        dt = torch.float8_e4m3fn
        try:
            # Test if the dtype is supported on this CUDA version
            _ = torch.tensor([0.0], device=DEVICE, dtype=dt)
            return dt
        except Exception as e:
            pass
    return torch.bfloat16

FP8_DTYPE = _pick_fp8_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min

_MXFP4_CODEBOOK = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25,
        0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=DEVICE
)   

def quantize_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    fp8_tensor = (tensor / scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)

def _encode_e8m0_pow2(scale_f32: torch.Tensor) -> torch.Tensor:
    log2_scale = torch.log2(scale_f32.clamp(min=2.0 ** -126))
    exp_unbiased = torch.round(log2_scale).to(torch.int32)
    exp_biased = (exp_unbiased + 127).clamp(0, 255)
    return exp_biased.to(torch.uint8)

def e8m0_to_f32(scale_e8m0: torch.Tensor) -> torch.Tensor:
    exp_unbiased = scale_e8m0.to(torch.int32) - 127
    return torch.pow(2.0, exp_unbiased.to(torch.float32))

def dynamic_mxfp4_quant(x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x2d.shape
    nb = cols // MXFP4_BLOCK
    x_blk = x2d.view(rows, nb, MXFP4_BLOCK).to(torch.float32)
    amax = x_blk.abs().amax(dim=-1).clamp(min=1e-12)
    block_scales = (amax / MXFP4_E2M1_MAX).clamp(min=2.0 ** -126)
    scale_e8m0 = _encode_e8m0_pow2(block_scales)
    scaled_decoded = e8m0_to_f32(scale_e8m0)
    x_norm = x_blk / scaled_decoded.unsqueeze(-1)
    dist = (x_norm.unsqueeze(-1) - _MXFP4_CODEBOOK.view(1, 1, 1, 16)).abs()
    q_idx = dist.argmin(dim=-1).to(torch.uint8)
    q_flat = q_idx.view(rows, cols)
    lo = q_flat[:, 0::2]
    hi = q_flat[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed, scale_e8m0.contiguous()

def mxfp4_to_f32(fp4_data_2d: torch.Tensor) -> torch.Tensor:
    byte = fp4_data_2d.to(torch.uint8)
    lo = byte & 0x0F
    hi = (byte >> 4) & 0x0F
    rows, packed_cols = fp4_data_2d.shape
    unpacked_cols = packed_cols * 2
    out_idx = torch.empty((rows, unpacked_cols), dtype=torch.uint8, device=fp4_data_2d.device)
    out_idx[:, 0::2] = lo
    out_idx[:, 1::2] = hi
    return _MXFP4_CODEBOOK[out_idx.long()]

def quantize_mxfp4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b,m,n = tensor.shape
    fp4_data_2d, scale_e8m0 = dynamic_mxfp4_quant(tensor.reshape(b * m, n))
    return fp4_data_2d.view(b, m, n // 2), scale_e8m0

def dequantize_mxfp4_block(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: Tuple[int, int, int],
) -> torch.Tensor:
    b,m,n = orig_shape
    num_rows = b * m
    num_blocks = n // MXFP4_BLOCK
    fp4_data_2d = fp4_data.reshape(num_rows, n // 2)
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]
    float_vals_blocked = float_vals.view(num_rows, num_blocks, MXFP4_BLOCK)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)
    return scaled.reshape(b, m, n).to(torch.bfloat16)

def _mla_decode_splitk_pytorch(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: Dict,
    num_kv_splits: int = 16,
) -> torch.Tensor:
    nq = config["num_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    batch = config["batch_size"]

    total_q_tokens = q.shape[0]
    out = torch.empty((total_q_tokens, nq, dv), dtype=torch.bfloat16, device=DEVICE)

    for b in range(batch):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        k_start = int(kv_indptr[b].item())
        k_end = int(kv_indptr[b + 1].item())

        tq = q_end - q_start
        tk = k_end - k_start
        if tq == 0 or tk == 0:
            if tq > 0:
                out[q_start:q_end] = 0.0
            continue
        
        q_b = q[q_start:q_end].to(torch.float32)
        kv_b = kv_buffer[k_start:k_end]

        k_flat = kv_b.squeeze(1).to(torch.float32)
        v_flat = k_flat[:, :dv]

        split_size = max(1, (tk + num_kv_splits - 1) // num_kv_splits)
        actual_splits = (tk + split_size - 1) // split_size

        partial_o = torch.zeros((actual_splits, tq, nq, dv), dtype=torch.float32, device=DEVICE)
        partial_m = torch.full((actual_splits, tq, nq), float('-inf'), dtype=torch.float32, device=DEVICE)
        partial_l = torch.zeros((actual_splits, tq, nq), dtype=torch.float32, device=DEVICE)

        for s in range(actual_splits):
            ks = s * split_size
            ke = min(ks + split_size, tk)

            k_chunk = k_flat[ks:ke]
            score = torch.einsum('qhd,kd->qhk', q_b, k_chunk) * sm_scale

            m_s = score.amax(dim=-1)
            p = torch.exp(score - m_s.unsqueeze(-1))
            l_s = p.sum(dim=-1)

            v_chunk = v_flat[ks:ke]
            o_s = torch.einsum('qhk,kd->qhd', p, v_chunk)

            partial_o[s] = o_s
            partial_m[s] = m_s
            partial_l[s] = l_s
        
        global_m = partial_m.amax(dim=0)

        alpha = torch.exp(partial_m - global_m.unsqueeze(0))
        corrected_l = (partial_l * alpha).sum(dim=0)
        corrected_o = (partial_o * alpha.unsqueeze(-1)).sum(dim=0)

        result = corrected_o / corrected_l.clamp(min=1e-12).unsqueeze(-1)
        out[q_start:q_end] = result.to(torch.bfloat16)
    
    return out

def _pick_num_splits(kv_seq_len: int, batch_size: int) -> int:
    if batch_size >= 256:
        if kv_seq_len <= 1024:
            return 4
        else:
            return 8
    elif batch_size >= 64:
        if kv_seq_len <= 1024:
            return 4
        elif kv_seq_len <= 4096:
            return 8
        else:
            return 16
    elif batch_size >= 32:
        if kv_seq_len <= 1024:
            return 8
        else:
            return 16
    else:
        if kv_seq_len <= 1024:
            return 32
        else:
            return 64

def generate_input(
    batch_size: int, q_seq_len: int, kv_seq_len: int, seed: int
) -> input_t:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    total_q = batch_size * q_seq_len
    total_kv = batch_size * kv_seq_len

    q = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )
    kv_buffer_bf16 = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )

    kv_buffer_fp8, kv_scale_fp8 = quantize_fp8(kv_buffer_bf16)
    kv_buffer_mxfp4, kv_scale_mxfp4 = quantize_mxfp4(kv_buffer_bf16)

    kv_data = {
        "bf16": kv_buffer_bf16,
        "fp8": (kv_buffer_fp8, kv_scale_fp8),
        "mxfp4": (kv_buffer_mxfp4, kv_scale_mxfp4),
    }

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=DEVICE) * q_seq_len
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=DEVICE) * kv_seq_len

    config = {
        "batch_size": batch_size,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "qk_head_dim": QK_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "q_seq_len": q_seq_len,
        "kv_seq_len": kv_seq_len,
        "sm_scale": SM_SCALE,
    }

    return (q, kv_data, qo_indptr, kv_indptr, config)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    if "mxfp4" in kv_data:
        kv_fp4, kv_scale_e8m0 = kv_data["mxfp4"]
        total_kv = kv_fp4.shape[0]
        kv_buffer = dequantize_mxfp4_block(
            kv_fp4, kv_scale_e8m0,
            (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        )
    elif "fp8" in kv_data:
        kv_fp8, kv_scale = kv_data["fp8"]
        kv_buffer = kv_fp8.to(torch.float32) * kv_scale
        kv_buffer = kv_buffer.to(torch.bfloat16)
    else:
        kv_buffer = kv_data["bf16"]

    num_splits = _pick_num_splits(kv_seq_len, batch_size)

    return _mla_decode_splitk_pytorch(
        q=q, kv_buffer=kv_buffer,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        config=config, num_kv_splits=num_splits,
    )

check_implementation = make_match_reference(custom_kernel, rtol=1e-1, atol=1e-1)

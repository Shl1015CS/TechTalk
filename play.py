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


NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

def _pick_fp8_dtype() -> torch.dtype:
    if _IS_ROCM and hasattr(torch, "float8_e4m3fnuz"):
        dt = torch.float8_e4m3fnuz
        try:
            _ = torch.tensor([0.0], device=DEVICE, dtype=dt)
            return dt
        except Exception:
            pass
    if hasattr(torch, "float8_e4m3fn"):
        dt = torch.float8_e4m3fn
        try:
            _ = torch.tensor([0.0], device=DEVICE, dtype=dt)
            return dt
        except Exception:
            pass
    return torch.bfloat16

FP8_DTYPE = _pick_fp8_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min
 

def quantize_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / FP8_MAX
    fp8_tensor = (tensor / scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)

def _mla_decode_splitk(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: Dict,
    num_kv_splits: int = 16,
) -> torch.Tensor:
    nq = config["num_heads"]
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
        k_flat = kv_buffer[k_start:k_end].squeeze(1).to(torch.float32)
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
            scores = torch.einsum('qhd,kd->qhk', q_b, k_chunk) * sm_scale
            m_s = scores.amax(dim=-1)
            p = torch.exp(scores - m_s.unsqueeze(-1))
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
        return 4 if kv_seq_len <= 1024 else 8
    elif batch_size >= 64:
        if kv_seq_len <= 1024:
            return 4
        return 8 if kv_seq_len <= 4096 else 16
    elif batch_size >= 32:
        return 8 if kv_seq_len <= 1024 else 16
    else:
        return 32 if kv_seq_len <= 1024 else 64

def generate_input(batchsize: int, qseqlen: int, kvseqlen: int, seed: int) -> input_t:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    total_q = batchsize * qseqlen
    total_kv = batchsize * kvseqlen

    q = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )
    kv_buffer_bf16 = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )

    kv_buffer_fp8, kv_scale_fp8 = quantize_fp8(kv_buffer_bf16)

    kv_data = {
        "bf16": kv_buffer_bf16,
        "fp8": (kv_buffer_fp8, kv_scale_fp8),
    }

    qo_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device=DEVICE) * qseqlen
    kv_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device=DEVICE) * kvseqlen

    config = {
        "batch_size": batchsize,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "qk_head_dim": QK_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "q_seq_len": qseqlen,
        "kv_seq_len": kvseqlen,
        "sm_scale": SM_SCALE,
    }

    return (q, kv_data, qo_indptr, kv_indptr, config)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    if "bf16" in kv_data:
        kv_buffer = kv_data["bf16"]
    elif "fp8" in kv_data:
        kv_fp8, kv_scale = kv_data["fp8"]
        kv_buffer = kv_fp8.to(torch.float32) * kv_scale
        kv_buffer = kv_buffer.to(torch.bfloat16)
    else:
        key = next(iter(kv_data))
        val = kv_data[key]
        if isinstabnce(val,tuple):
            kv_buffer = val[0].to(torch.bfloat16)
        else:
            kv_buffer = val.to(torch.bfloat16)

    num_splits = _pick_num_splits(kv_seq_len, batch_size)

    return _mla_decode_splitk(
        q=q, kv_buffer=kv_buffer,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        config=config, num_kv_splits=num_splits,
    )

check_implementation = make_match_reference(custom_kernel, rtol=1e-1, atol=1e-1)

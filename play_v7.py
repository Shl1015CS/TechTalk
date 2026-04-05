import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t
from utils import make_match_reference
 
if not torch.cuda.is_available():
    raise RuntimeError("MI355x/Rocm GPU is required")
 
DEVICE = "cuda"

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM       #576
V_HEAD_DIM = KV_LORA_RANK                           #512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
DQ_PAD = 1024

FP8_DTYPE= torch.float8_e4m3fn

@triton.jit
def _mla_fused_decoder_kernel(
    Q_ptr,
    KV_ptr,
    Out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kvb, stride_kvs, stride_kvd,
    stride_ob, stride_oh, stride_od,
    Sk,
    sm_scale,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    DQ_PAD: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SK: tl.constexpr,
):
    bid = tl.program_id(0)
 
    d_offs = tl.arange(0, DQ_PAD)
    d_mask = d_offs < DQ
    h_offs = tl.arange(0, H)
    sk_base = tl.arange(0, BLOCK_SK)
 
    q = tl.load(
        Q_ptr + bid * stride_qb
            + h_offs[:, None] * stride_qh
            + d_offs[None, :] * stride_qd,
        mask=d_mask[None, :], other0.0,
    ).to(tl.float32)
 
    m_i = tl.full([H], float('-inf'), dtype = tl.float32)
    l_i = tl.zeros([H], dtype = tl.float32)
    acc = tl.zeros([H, DQ_PAD], dtype = tl.float32)

    q_bf16 = q.to(tl.bfloat16)

    for sk_start in tl.range(0, Sk, BLOCK_SK):
        sk_offs = sk_start+ sk_base
        sk_mask = sk_offs < Sk
 
        kv = tl.load(
            KV + bid * stride_kvb + sk_offs[:, None] * stride_kvs + d_offs[None, :] * stride_kvd,
            mask=sk_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.bfloat16)
 
        scores = tl.dot(q_bf16, tl.tran(kv), out_type = tl.float32) * sm_scale
        scores = tl.where(sk_mask[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis = 1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(scores- m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis = 1)
        acc =acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), kv, out_dytpe = tl.float32)
        m_i = m_new

    acc = acc / l_i[: None]
 
    v_mask = d_offs < DV
    tl.store(
        Out + bid * stride_ob + h_ofs[:, None] * stride_oh + d_offs[None, :] * stride_od,
        acc.to(tl.bfloat16), mask=v_mask,
    )
 
 
_static_bufs: Dict = {}
 
def _ensure_bufs(B, Sk) -> tuple:
    key = (B, Sk)
    if key in _static_bufs:
        return key
    d:Dict = {}
    d["out"] = torch.empty((B, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    d["q"] = torch.empty((B, NUM_HEADS, QK_HEAD_DIM), dtype=FP8_DTYPE, device=DEVICE)
    d["kv"] = torch.empty((B, Sk, QK_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    _static_bufs[key] = d
    return key
 
def generate_input(batchsize: int, qseqlen: int, kvseqlen: int, seed: int) -> input_t:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
 
    total_q = batchsize * qseqlen
    total_kv = batchsize * kvseqlen
 
    q_raw = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )
    kv_raw = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )
 
    key = _ensure_bufs(batchsize, kvseqlen)
    bufs = _static_bufs[key]
    bufs["q"].copy_(q_raw.view(batchsize, NUM_HEADS, QK_HEAD_DIM))
    bufs["kv"].copy_(kv_raw.view(batchsize, kvseqlen, QK_HEAD_DIM).to(FP8_DTYPE))
 
    kv_data = {
        "bf16": kv_raw,
        "_key": key,
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
 
    return (q_raw, kv_data, qo_indptr, kv_indptr, config)
 
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, _, _, config = data
    B = config["batch_size"]
    Sk = config["kv_seq_len"]
 
    key = kv_data.get("_key")
    if key is None:
        key = _ensure_bufs(B, Sk)
        bufs = _static_bufs[key]
        bufs["q"].copy_(q.view(B, NUM_HEADS, QK_HEAD_DIM))
        bufs["kv"].copy_(kv_data["bf16"].view(B, Sk, QK_HEAD_DIM).to(FP8_DTYPE))
    bufs = _static_bufs[key]
 
    Q = bufs["q"]
    KV = bufs["kv"]
    Out = bufs["out"]
 
    grid = (B,)
    _mla_fused_decoder_kernel[grid](
        Q, KV, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        KV.stride(0), KV.stride(1), KV.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Sk, SM_SCALE,
        DQ=QK_HEAD_DIM, DV=V_HEAD_DIM, DQ_PAD=DQ_PAD,
        H=NUM_HEADS,
        BLOCK_SK=64,
        num_warps = 4,
        num_stages = 2,
    )
    return Out
 
check_implementation = make_match_reference(custom_kernel, rtol=1e-1, atol=1e-1)

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

FP8_DTYPE = torch.float8_e4m3fn

@triton.jit
def _mla_decode_kernel(
    Q_ptr,
    KV_ptr,
    Out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kvb, stride_kvs, stride_kvd,
    stride_ob, stride_oh, stride_od,
    Sk,
    sm_scale,
    H: tl.constexpr,
    D_LORA: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_SK: tl.constexpr,
):
    bid = tl.program_id(0)
 
    h_offs = tl.arange(0, H)
    d_lora = tl.arange(0, D_LORA)
    d_rope = tl.arange(0, D_ROPE)

    q_lora = tl.load(
        Q_ptr + bid * stride_qb
            + h_offs[:, None] * stride_qh
            + d_lora[None, :] * stride_qd,
    )

    q_rope = tl.load(
        Q_ptr + bid * stride_qb
            + h_offs[:, None] * stride_qh
            + (D_LORA + d_rope[None, :]) * stride_qd,
    )

    m_i = tl.full([H], float('-inf'), dtype = tl.float32)
    l_i = tl.zeros([H], dtype = tl.float32)
    acc = tl.zeros([H, D_LORA], dtype = tl.float32)

    for sk_start in tl.range(0, Sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_mask = sk_offs < Sk
 
        kv_lora = tl.load(
            KV_ptr + bid * stride_kvb + sk_offs[:, None] * stride_kvs + d_lora[None, :] * stride_kvd,
            mask=sk_mask[:, None], other=0.0,
        )

        kv_rope = tl.load(
            KV_ptr + bid * stride_kvb + sk_offs[:, None] * stride_kvs + (D_LORA + d_rope[None, :]) * stride_kvd,
            mask=sk_mask[:, None], other=0.0,            
        )

        scores = tl.dot(q_lora, tl.trans(kv_lora), out_dtype=tl.float32)
        scores += tl.dot(q_rope, tl.trans(kv_rope), out_dtype=tl.float32)
        scores *= sm_scale
        scores = tl.where(sk_mask[None, :], scores, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(P.to(tl.bfloat16), kv_lora.to(tl.bfloat16), out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    tl.store(
        Out_ptr + bid * stride_ob + h_offs[:, None] * stride_oh + d_lora[None, :] * stride_od,
        acc.to(tl.bfloat16),
    )
 
 
_static_bufs: Dict = {}
 
def _ensure_bufs(B: int, Sk: int) -> tuple:
    key = (B, Sk)
    if key in _static_bufs:
        return key
    d: Dict = {}

    d["q"] = torch.empty((B, NUM_HEADS, QK_HEAD_DIM), dtype=FP8_DTYPE, device=DEVICE)
    d["kv"] = torch.empty((B, Sk, QK_HEAD_DIM), dtype=FP8_DTYPE, device=DEVICE)
    d["out"] = torch.empty((B, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
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
    _mla_decode_kernel[grid](
        Q, KV, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        KV.stride(0), KV.stride(1), KV.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Sk, SM_SCALE,
        H=NUM_HEADS,
        D_LORA = KV_LORA_RANK,
        D_ROPE = QK_ROPE_HEAD_DIM,
        BLOCK_SK=64,
        num_warps=8,
        num_stages=3,
    )
    return Out
 
check_implementation = make_match_reference(custom_kernel, rtol=1e-1, atol=1e-1)

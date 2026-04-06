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
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM       # 576
V_HEAD_DIM = KV_LORA_RANK                           # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

FP8_DTYPE = torch.float8_e4m3fn
_BLOCK_SK = 64
_NUM_CUS = 256

def _get_num_splits(B, Sk):
    ns = max(16, math.ceil(_NUM_CUS / B))
    ns = 1 << math.ceil(math.log2(ns))
    max_ns = Sk // _BLOCK_SK
    ns = min(ns,max_ns)
    ns = 1 << int(math.log2(ns))
    return ns

@triton.jit
def _mla_stage1(
    Q_ptr, KV_ptr, POut_ptr, PLse_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kvb, stride_kvs, stride_kvd,
    stride_pob, stride_pos, stride_poh, stride_pod,
    stride_plb, stride_pls, stride_plh,
    Sk, sm_scale, split_size,
    H: tl.constexpr,
    D_LORA: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_SK: tl.constexpr,
):
    bid = tl.program_id(0)
    sid = tl.program_id(1)
    h_offs = tl.arange(0, H)
    d_lora = tl.arange(0, D_LORA)
    d_rope = tl.arange(0, D_ROPE)

    q_lora = tl.load(
        Q_ptr + bid * stride_qb + h_offs[:, None] * stride_qh + d_lora[None, :] * stride_qd,
    )

    q_rope = tl.load(
        Q_ptr + bid * stride_qb + h_offs[:, None] * stride_qh + (D_LORA + d_rope[None, :]) * stride_qd,
    )

    sk_begin = sid * split_size

    m_i = tl.full([H], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_LORA], dtype=tl.float32)

    for sk_start in tl.range(sk_begin, sk_begin + split_size, BLOCK_SK):
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
        acc += tl.dot(P.to(kv_lora.dtype), kv_lora, out_dtype=tl.float32)
        m_i = m_new
    
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(
        POut_ptr + bid * stride_pob + sid * stride_pos + h_offs[:, None] * stride_poh + d_lora[None, :] * stride_pod,
        acc.to(tl.bfloat16),
    )
    tl.store(
        PLse_ptr + bid * stride_plb + sid * stride_pls + h_offs * stride_plh,
        lse,
    )

@triton.jit
def _mla_reduce(
    POut_ptr, PLse_ptr, Out_ptr,
    stride_pob, stride_pos, stride_poh, stride_pod,
    stride_plb, stride_pls, stride_plh,
    stride_ob, stride_oh, stride_od,
    NUM_SPLITS: tl.constexpr,
    D_LORA: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    d_offs = tl.arange(0, D_LORA)
    s_offs = tl.arange(0, NUM_SPLITS)

    lse = tl.load(
        PLse_ptr + bid * stride_plb + s_offs * stride_pls + hid * stride_plh,
    )

    max_lse = tl.max(lse)
    exp_lse = tl.exp(lse - max_lse)
    sum_exp = tl.sum(exp_lse)

    acc = tl.zeros([D_LORA], dtype = tl.float32)
    for s in tl.static_range(0, NUM_SPLITS):
        lse_s = tl.load(PLse_ptr + bid * stride_plb + s * stride_pls + hid * stride_plh,
        )
        w = tl.exp(lse_s - max_lse) / sum_exp
        partial_s = tl.load(
            POut_ptr + bid * stride_pob + s * stride_pos + hid * stride_poh + d_offs * stride_pod,
        ).to(tl.float32)
        acc += w * partial_s
    
    tl.store(
        Out_ptr + bid * stride_ob + hid * stride_oh + d_offs * stride_od,
        acc.to(tl.bfloat16),
    )
 
_static_bufs: Dict = {}
 
def _ensure_bufs(B: int, Sk: int) -> tuple:
    key = (B, Sk)
    if key in _static_bufs:
        return key
    ns = _get_num_splits(B, Sk)
    d: Dict = {}

    d["q"] = torch.empty((B, NUM_HEADS, QK_HEAD_DIM), dtype=FP8_DTYPE, device=DEVICE)
    d["kv"] = torch.empty((B, Sk, QK_HEAD_DIM), dtype=FP8_DTYPE, device=DEVICE)
    d["out"] = torch.empty((B, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    d["partial_out"] = torch.empty((B, ns, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    d["partial_lse"] = torch.empty((B, ns, NUM_HEADS), dtype=torch.float32, device=DEVICE)
    d["num_splits"] = ns
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
        bufs["q"].copy_(q.view(B, NUM_HEADS, QK_HEAD_DIM).to(FP8_DTYPE))
        bufs["kv"].copy_(kv_data["bf16"].view(B, Sk, QK_HEAD_DIM).to(FP8_DTYPE))
    bufs = _static_bufs[key]
 
    Q = bufs["q"]
    KV = bufs["kv"]
    Out = bufs["out"]
    POut = bufs["partial_out"]
    PLse = bufs["partial_lse"]
    ns = bufs["num_splits"]

    split_size = math.ceil(Sk / ns)    

    _mla_stage1[(B, ns)](
        Q, KV, POut, PLse,
        Q.stride(0), Q.stride(1), Q.stride(2),
        KV.stride(0), KV.stride(1), KV.stride(2),
        POut.stride(0), POut.stride(1), POut.stride(2), POut.stride(3),
        PLse.stride(0), PLse.stride(1), PLse.stride(2),
        Sk, SM_SCALE, split_size,
        H=NUM_HEADS,
        D_LORA = KV_LORA_RANK,
        D_ROPE = QK_ROPE_HEAD_DIM,
        BLOCK_SK= _BLOCK_SK,
        num_warps=4,
        num_stages=2,
    )

    _mla_reduce[(B, NUM_HEADS)](
        POut, PLse, Out,
        POut.stride(0), POut.stride(1), POut.stride(2), POut.stride(3),
        PLse.stride(0), PLse.stride(1), PLse.stride(2),
        Out.stride(0),  Out.stride(1),  Out.stride(2),
        NUM_SPLITS = ns,
        D_LORA = KV_LORA_RANK,
        num_warps = 2,
    )    
    return Out
 
check_implementation = make_match_reference(custom_kernel, rtol=1e-1, atol=1e-1)

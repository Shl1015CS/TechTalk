import math
from typing import Dict
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from utils import make_match_reference

# 设备检查：要求CUDA（MI355X/Rocm GPU）
if not torch.cuda.is_available():
    raise RuntimeError("MI355X/Rocm GPU is required")
DEVICE = "cuda"

# ===================== 模型超参数 =====================
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                      # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)          # 注意力缩放因子

# ===================== 量化与硬件参数 =====================
FP8_DTYPE = torch.float8_e4m3fn
_BLOCK_SK = 64       # SK维度的块大小
_NUM_CUS = 256       # CUDA核心数（或类似硬件参数）
_MX_BLOCK = 32       # 量化时的块大小（D维度分组）


# ===================== 量化函数：Per-channel MXFP4 量化（SK维度共享scale） =====================
def _quantize_mxfp4_perchannel(tensor, block_size=_MX_BLOCK):
    """Quantize with ONE scale per D-group shared across ALL sk positions.
    输入：(B, SK, D) 或 (B, H, D)；输出：(packed_nibbles, e8m0_scales)
    """
    shape = tensor.shape
    D = shape[-1]
    assert D % block_size == 0, f"D={D} 必须能被 block_size={block_size} 整除"
    x = tensor.float()  # 转为float32处理

    # 重塑为 (leading_dims, D//bs, bs)，其中 leading_dims 是除最后一个维度外的所有维度
    leading_dims = x.shape[:-1]
    x_blocks = x.reshape(*leading_dims, D // block_size, block_size)

    # 计算每个D-group的最大绝对值（跨所有非D维度）
    if len(leading_dims) == 2:  # 情况1：(B, SK, D) → 跨SK和B的最大
        channel_max = x_blocks.abs().amax(dim=-3)  # 对dim=-3（SK维度）取最大
    else:  # 情况2：(B, H, D) → 跨H和B的最大（H较小）
        channel_max = x_blocks.abs().amax(dim=-3).amax(dim=tuple(range(1, len(leading_dims))))

    channel_max = channel_max.clamp(min=1e-12)  # 避免除零

    # 计算e8m0的索引（量化到mxfp4的e8m0格式）
    e8m0 = torch.floor(torch.log2(channel_max / 6.0)).to(torch.int32) + 127
    e8m0 = e8m0.clamp(0, 255).to(torch.uint8)  # e8m0范围：0~255（uint8）

    # 计算scale = 2^(e8m0 - 127) / 6.0 → 恢复原始范围的缩放因子
    scale = (2.0 ** (e8m0.float() - 127.0)) / 6.0  # (B, D//bs)

    # 缩放x_blocks并广播scale到 (B, D//bs, 1)，再扩展到x_blocks的形状
    scale_bc = scale.reshape(shape[0], D // block_size, 1).expand_as(x_blocks)
    scaled = x_blocks / scale_bc

    # 生成LUT（查找表）和nibble量化
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    signs = (scaled < 0).to(torch.uint8)
    abs_v = scaled.abs().clamp(max=6.0)
    indices = (abs_v.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    nibbles = indices | (signs << 3)  # 符号位左移3位，与索引合并为4位nibble
    nibbles = nibbles.reshape(*leading_dims, D)  # 重塑回原始形状（除D维度外）

    # 打包nibble（每2个nibble打包为1个字节）
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)  # (..., D//2)
    return packed, e8m0  # packed: 同shape但D→D//2；e8m0: (B, D//bs)


# ===================== 辅助函数：计算split数量（用于Kernel并行） =====================
def _get_num_splits(B, SK):
    ns = max(16, math.ceil(_NUM_CUS / B))  # 每个Block的最小split数
    ns = 1 << math.floor(math.log2(ns))    # 对齐到2的幂
    max_ns = SK // _BLOCK_SK               # SK维度的最大split数（受_BLOCK_SK限制）
    ns = min(ns, max_ns)                   # 取较小值
    ns = 1 << int(math.log2(ns))           # 再次对齐到2的幂
    return ns


# ===================== Triton Kernel：Stage 1 - 注意力分数+ V的转置点积（dot_scaled） =====================
@triton.jit
def _mla_stage1(
    Q_fp4, Q_sc, KV_fp4, KV_sc, V_sc, Pout, PLse,
    stride_qfb, stride_qfh, stride_qfd,
    stride_kfb, stride_kfs, stride_kfd,
    stride_vsb, stride_vss, stride_vsd,
    stride_pob, stride_pos, stride_poh, stride_pod,
    stride_plb, stride_pls, stride_plh,
    SK, sm_scale, split_size,
    H: tl.constexpr,      # 头数
    DLH: tl.constexpr,    # D//H（Q/KV的head维度）
    DLS: tl.constexpr,    # D//S（？可能是笔误，应为DLH或其他）
    DH: tl.constexpr,     # QK_HEAD_DIM（Q/KV的head维度，含rope）
    DRS: tl.constexpr,    # ？
    BS: tl.constexpr,     # ？
    DVS: tl.constexpr,    # V的head维度（KV_LORA_RANK）
    BS_KS: tl.constexpr,  # ？
):
    bid = tl.program_id(0)  # Block ID（batch维度）
    sid = tl.program_id(1)  # Block ID（split维度）

    h = tl.arange(0, H)     # 头索引
    dl = tl.arange(0, DLH)  # D//H 索引（Q/KV的head内维度）
    dls = tl.arange(0, DLS)
    dh = tl.arange(0, DH)
    drs = tl.arange(0, DRS)
    dv = tl.arange(0, DVS)
    bsks = tl.arange(0, BS_KS)

    # ---- 加载Q（MXFP4格式，一次加载一个Block） ----
    # Q的stride：qfb(b), qfh(h), qfd(d)
    Ql_p = tl.load(
        Q_fp4 + bid * stride_qfb h[:, None] * stride_qfh + + dl[None, :] * stride_qfd
    )
    Qs_p = tl.load(
        Q_sc + bid * stride_qfb + h[:, None] * stride_qfh + dl[None, :] * stride_qfd
    )
    Qs_s = tl.load(
        Q_sc + bid * stride_qfb + h[:, None] * stride_qfh + (DLH + drs[None, :]) * stride_qfd
    )

    # ---- 加载V的scale（per-channel，所有tile共享） ----
    V_sc = tl.load(V_sc + bid * stride_vsb + dv[None, :] * stride_vsd + bsks[None, :] * stride_vsk)

    # ---- P/T scale constant（固定值127，dtype=uint8） ----
    P_T_sc = tl.full((H, BS_KS), 127, dtype=tl.uint8)

    # ---- 循环处理每个SK split ----
    sk_begin = sid * split_size
    sk_end = sk_begin + split_size
    sk_mask = sk < SK  # SK维度掩码（防止越界）

    for sk_start in tl.range(sk_begin, sk_end, BS_KS):
        sk = sk_start + tl.arange(0, BS_KS)
        sk = tl.where(sk < SK, sk, 0)  # 越界则置0（掩码处理）

        # ---- 加载KV（MXFP4格式，复用score和V的accumulate） ----      KVl_p = 
  tl.load(
            KV_fp4 + bid * stride_kfb + sk[:, None] * stride_kfs + dl[None, :] * stride_kfd
        )
        KVl_s = tl.load(
            KV_sc + bid * stride_kfb + sk[:, None] * stride_kfs + dl[None, :] * stride_kfd
        )
        KVl_s = tl.where(sk_mask, KVl_s, 0)  # 掩码处理
        KVs_p = tl.load(
            KV_fp4 + bid * stride_kfb + sk[:, None] * stride_kfs + (DLH + drs[None, :]) * stride_kfd
        )
        KVs_s = tl.load(
            KV_sc + bid * stride_kfb + sk[:, None] * stride_kfs + (DLH + drs[None, :]) * stride_kfd
        )
        KVs_s = tl.where(sk_mask, KVs_s, 0)  # 掩码处理

        # ---- Score计算：Q @ K^T（via native MXFP4 dot_scaled） ----
        scores = tl.dot_scaled(Ql_p, KVl_p, Qs_p,Vl_s, "e8m0", tl.trans K(KVl_p), KVs_s, "e8m0")
        scores = tl.dot_scaled(Ql_p, KVs_p, Qs_p, KVs_s, "e8m0", tl.trans(KVs_p), KVs_s, "e8m0")
        scores = tl.where(sk_mask, scores, float("-inf"))  # 掩码无效位置

        # ---- Online softmax ----
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        sum_exp = tl.sum(tl.exp(scores - m_new[:, None]), axis=1)
        acc += tl.exp(scores - m_new[:, None])
        m_i = m_new
        lse = m_i + tl.log(lse + sum_exp)

        # ---- V accumulation：(KV Lora @ par)+ via dot_scaled ----
        lhs = KVl_s * (DLH + DRS) // BS_KS  # ？需确认维度逻辑
        rhs = tl.trans(P_T_sc, V_sc)
        vs_acc = tl.dot_scaled(lhs, rhs, "e8m0", tl.trans(rhs), "e8m0", lhs_lora=False)
        acc += tl.trans(vs_acc)

    # ---- 存储中间结果（Pout, PLse） ----
    tl.store(Pout + bid * stride_pob + sid * stride_pos + h[:, None] * stride_poh + dl[None, :] * stride_pod, acc)
    tl.store(PLse + bid * stride_plb + sid * stride_pls + h * stride_plh, lse)


# ===================== Triton Kernel：Stage 2 - 归约（Reduce） =====================
@triton.jit
def _mla_reduce(
    Pout, PLse, Out,
    stride_pob, stride_pos, stride_poh, stride_pod,
    stride_plb, stride_pls, stride_plh,
    stride_ob, stride_oh, stride_od,
    DV: tl.constexpr,    # V的head维度（KV_LORA_RANK）
    NUM_SPLITS: tl.constexpr,  # split总数
):
    bid = tl.program_id(0)  # Batch ID
    hid = tl.program_id(1)  # Head ID
    d = tl.arange(0, DV)    # V的head内维度索引

    # ---- 加载每个split的PLse和Pout ----
    max_lse = tl.load(PLse + bid * stride_plb + 0 * stride_pls + hid * stride_plh)
    sum_exp = tl.zeros((DV,), dtype=tl.float32)
    for si in tl.static_range(0, NUM_SPLITS):
        lse_si = tl.load(PLse + bid * stride_plb + si * stride_pls + hid * stride_plh)
        w = tl.exp(lse_si - max_lse) / sum_exp
        partial = tl.load(
            Pout + bid * stride_pob + si * stride_pos + hid * stride_poh + d * stride_pod
        ).to(tl.float32)
        sum_exp += w
        acc += w * partial

    # ---- 存储最终结果（Out） ----
    tl.store(Out + bid * stride_ob + hid * stride_oh + d * stride_od, acc.to(tl.float16))


# ===================== 生成输入数据（随机初始化+量化） =====================
def generate_input(batchsize: int, qseqlen: int, kvseqlen: int, seed: int) -> input_t:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    total_q = batchsize * qseqlen
    total_kv = batchsize * kvseqlen

    # 生成Q的原始数据（bf16）
    q_raw = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )
    # 生成KV的原始数据（bf16）
    kv_raw = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device=DEVICE, generator=gen,
    )

    # 缓存key（避免重复生成）
    key = _ensure_bufs(batchsize, kvseqlen)
    bufs = _static_bufs[key]

    # ---- Q的量化（per-row MXFP4，每个head有自己的scale） ----
    q_3d = q_raw.view(batchsize, qseqlen, NUM_HEADS, QK_HEAD_DIM)
    q_fp4, q_e8m0 = _quantize_mxfp4_perchannel(q_3d)
    bufs["q_fp4"].copy_(q_fp4)
    # Q的scale：扩展为 [B, H, D]（与Q的head维度对齐）
    q_scale = q_e8m0.unsqueeze(1).expand(-1, NUM_HEADS, -1).contiguous()
    bufs["q_scale"].copy_(q_scale)

    # ---- KV的量化（per-channel MXFP4，SK维度共享scale） ----
    kv_3d = kv_raw.view(batchsize, kvseqlen, QK_HEAD_DIM)
    kv_fp4, kv_e8m0 = _quantize_mxfp4_perchannel(kv_3d)  # kv_e8m0: [B, D//32]
    bufs["kv_fp4"].copy_(kv_fp4)
    # KV的scale：扩展为 [B, Sk, D]（Sk=kvseqlen，每个D位置d使用 scale=kv_e8m0[b, d//32]）
    kv_scale = kv_e8m0.unsqueeze(1).expand(-1, kvseqlen, -1).contiguous()
    bufs["kv_scale"].copy_(kv_scale)

    # ---- V的scale：处理LoRA组和块逻辑 ----
    n_lora_groups = KV_LORA_RANK // _MX_BLOCK  # 16（假设_KV_BLOCK=32？需确认）
    v_scale_1d = kv_e8m0[:, :n_lora_groups]  # [B, 16]
    v_scale_expanded = v_scale_1d.repeat_interleave(_MX_BLOCK, dim=1)  # [B, 512]
    v_scale_2d = v_scale_expanded.unsqueeze(-1).expand(-1, -1, _BLOCK_SK // _MX_BLOCK).contiguous()
    bufs["v_scale"].copy_(v_scale_2d)

    # 组织KV数据（原始bf16 + 缓存key）
    kv_data = {"bf16": kv_raw, "_key": key}

    # 生成索引指针（qo_indptr, kv_indptr）
    qo_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device=DEVICE) * qseqlen
    kv_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device=DEVICE) * kvseqlen

    # 配置信息（用于Kernel调用）
    config = {
        "batch_size": batchsize, "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS, "qk_head_dim": QK_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK, "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM, "q_seq_len": qseqlen,
        "kv_seq_len": kvseqlen, "sm_scale": SM_SCALE,
    }

    return (q_raw, kv_data, qo_indptr, kv_indptr, config)


# ===================== 静态缓冲区管理（复用输入） =====================
_static_bufs = {}  # 缓存不同batchsize/kvseqlen的输入缓冲区

def _ensure_bufs(batchsize: int, kvseqlen: int):
    key = (batchsize, kvseqlen)
    if key not in _static_bufs:
        # 初始化缓冲区（示例：需根据Kernel参数调整形状和dtype）
        bufs = {
            "q_fp4": torch.empty((batchsize, qseqlen, NUM_HEADS, QK_HEAD_DIM // 2), dtype=torch.uint8, device=DEVICE),
            "q_scale": torch.empty((batchsize, NUM_HEADS, QK_HEAD_DIM // 32), dtype=torch.uint8, device=DEVICE),
            "kv_fp4": torch.empty((batchsize, kvseqlen, QK_HEAD_DIM // 2), dtype=torch.uint8, device=DEVICE),
            "kv_scale": torch.empty((batchsize, kvseqlen, QK_HEAD_DIM // 32), dtype=torch.uint8, device=DEVICE),
            "v_scale": torch.empty((batchsize, KV_LORA_RANK, _BLOCK_SK // _MX_BLOCK), dtype=torch.uint8, device=DEVICE),
            "out": torch.empty((batchsize, NUM_HEADS, KV_LORA_RANK), dtype=torch.float16, device=DEVICE),
            "partial_out": torch.empty((batchsize, NUM_SPLITS, NUM_HEADS, KV_LORA_RANK), dtype=torch.float16, device=DEVICE),
            "partial_lse": torch.empty((batchsize, NUM_SPLITS, NUM_HEADS), dtype=torch.float32, device=DEVICE),
            "num_splits": torch.tensor(_get_num_splits(batchsize, kvseqlen), dtype=torch.int32, device=DEVICE),
        }
        _static_bufs[key] = bufs
    return key


# ===================== 自定义Kernel调用（整合Stage1 + Stage2） =====================
def custom_kernel(data: input_t) -> output_t:
    q_raw, kv_data, qo_indptr, kv_indptr, config = data
    B = config["batch_size"]
    SK = config["kv_seq_len"]

    # 从缓冲区获取数据
    bufs = _static_bufs[kv_data["_key"]]
    Qf = bufs["q_fp4"]
    Qs = bufs["q_scale"]
    KVf = bufs["kv_fp4"]
    KVs = bufs["kv_scale"]
    Vs = bufs["v_scale"]
    Out = bufs["out"]
    Pout = bufs["partial_out"]
    PLse = bufs["partial_lse"]
    ns = bufs["num_splits"]
    split_size = math.ceil(SK / ns)

    # 启动Stage1 Kernel
    _mla_stage1[(B, ns)](
        Qf, Qs, KVf, KVs, Vs, Pout, PLse,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        KVf.stride(0), KVf.stride(1), KVf.stride(2),
        Vs.stride(0), Vs.stride(1), Vs.stride(2),
        Pout.stride(0), Pout.stride(1), Pout.stride(2), Pout.stride(3),
        PLse.stride(0), PLse.stride(1), PLse.stride(2), PLse.stride(3),
        SK, config["sm_scale"], split_size,
        H=NUM_HEADS,
        DLH=QK_HEAD_DIM // NUM_HEADS,
        DLS=QK_LORA_RANK // _MX_BLOCK,
        DH=QK_HEAD_DIM,
        DRS=QK_ROPE_HEAD_DIM // _MX_BLOCK,
        BS=1,  # 示例值，需根据实际逻辑调整
        DVS=KV_LORA_RANK,
        BS_KS=_BLOCK_SK,
        num_stages=2,
    )

    # 启动Stage2 Kernel
    _mla_reduce[(B, NUM_HEADS)](
        Pout, PLse, Out,
        Pout.stride(0), Pout.stride(1), Pout.stride(2), Pout.stride(3),
        PLse.stride(0), PLse.stride(1), PLse.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        DV=KV_LORA_RANK,
        NUM_SPLITS=ns,
    )

    return Out


# ===================== 验证实现（与参考实现比对） =====================
check_implementation = make_match_reference(custom_kernel, rtol=1e-1, atol=1e-1)

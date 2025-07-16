# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.1.0.dev0.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)

# Features not supported yet:
# - split (i.e. FlashDecoding)
# - tuned block sizes
# - paged KV
# - append KV to existing KV cache
# - FP8
# - bwd pass optimized for Hopper/Blackwell

import math
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute import utils
from flash_attn.cute.nsa.flash_nsa_fwd import NsaSlcForwardSm90


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _nsa_slc_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    topk: Optional[torch.Tensor] = None, #[head_k, total_group, topk]
    cu_group_q: Optional[torch.Tensor] = None, #[b + 1]
    group_size: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 64,
    num_threads: int = 384,
    _compute_capability: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = None
    total_q = q.shape[0]

    seqlen_k, num_head_kv, _ = k.shape[-3:]
    head_dim_v = v.shape[-1]
    
    assert k.shape == (seqlen_k, num_head_kv, head_dim)
    assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
    assert cu_seqlens_k.shape == (batch_size + 1,), "cu_seqlens_k must have shape (batch_size + 1,)"


    
    assert cu_seqlens_q.shape == (batch_size + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
            assert t.stride(0) == 1, "cu_seqlens_q, cu_seqlens_k must be contiguous"
    assert all(t is None or t.is_cuda for t in (q, k, v, cu_seqlens_q, cu_seqlens_k)), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qhead_per_kvhead = num_head // num_head_kv

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    out = torch.empty(*q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if requires_grad else None

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        utils.convert_from_dlpack(
            t.detach(), leading_dim=t.ndim - 1, divisibility=128 // dtype.width
        ) for t in (q, k, v, out)
    ]
    lse_tensor = utils.convert_from_dlpack(lse, leading_dim=lse.ndim - 1, alignment=4) if lse is not None else None
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None
        for t in (cu_seqlens_q, cu_seqlens_k,)
    ]
    
    compute_capability = torch.cuda.get_device_capability()[0] if _compute_capability is None else _compute_capability
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim
        if not causal:
            n_block_size = 192

    compile_key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, 
        lse is None, m_block_size, n_block_size, num_threads,
        compute_capability,
    )
    if compile_key not in _nsa_slc_fwd.compile_cache:
        if compute_capability == 9:
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = NsaSlcForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=False,
                pack_gqa=True,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
            )
        else:
            raise ValueError(f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x")
        # TODO: check @can_implement
        _nsa_slc_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, softmax_scale, current_stream,
            cu_seqlens_q_tensor, cu_seqlens_k_tensor,
        )
    _nsa_slc_fwd.compile_cache[compile_key](
        q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, softmax_scale, current_stream,
        cu_seqlens_q_tensor, cu_seqlens_k_tensor,
    )
    return out, lse


_nsa_slc_fwd.compile_cache = {}

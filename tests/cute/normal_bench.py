import torch
import math
import itertools
from datetime import datetime


from functools import partial
import torch.nn.functional as F

import pytest
import torch

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

# from padding import pad_input, unpad_input
#from flash_attn.utils.testing import attention_ref, generate_qkv, generate_random_padding_mask
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func, _flash_attn_fwd, _flash_attn_bwd
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward


def get_gpu_flops():
    name = gpu_name()
    if "A100" in name or "A800" in name:
        return 312e12
    elif "H20" in name:
        return 148e12
    elif "H100" in name or "H800" in name:
        return 989e12
    else:
        assert False, "unsupport gpu {}".format(name)

def gpu_name():
    id = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(id)
    return p.name

def get_gpu_bandwidth():
    name = gpu_name()
    if "A100" in name or "A800" in name:
        return 2e12
    elif "H20" in name:
        return 4e12
    elif "H100" in name or "H800" in name:
        return 3.35e12
    else:
        assert False, "unsupport gpu {}".format(name)

def sol_us(bsz, seq_q, seq_kv, head_q, head_kv, dim, is_causal, is_bwd = False, local_window_size = -1):
    assert seq_q == seq_kv, "sol only support self attn currently"
    is_mqa = (head_q > head_kv) and (seq_q == 1)
    q_mem = 2 * 2 * (bsz * seq_q * head_q * dim)
    kv_mem = 2 * (head_kv if is_mqa else head_q) * bsz * seq_kv * dim * 2
    mem = q_mem + kv_mem
    if is_bwd:
        mem = mem + q_mem

    bandwidth = get_gpu_bandwidth()

    mem_sol = (mem / bandwidth) * 1000 * 1000

    # default: causal, f16
    computation = 2*bsz*head_q * seq_q * seq_kv * dim
    if not is_causal:
        computation = computation * 2
    elif local_window_size > 0 and local_window_size < seq_q:
        # see how to get mfu on local window (https://bytedance.larkoffice.com/docx/VJTYdSXEHoPHLQxfBk9cWX5KnLd)
        computation = computation * ((2*seq_q * local_window_size - local_window_size * local_window_size) / (seq_q * seq_q))
    
    if is_bwd:
        computation = 2 * computation
    th_flops = get_gpu_flops()
    com_sol = (computation / th_flops) * 1000 * 1000

    bound_type = "compute bound" if com_sol > mem_sol else "memory bound"
    sol = com_sol if com_sol > mem_sol else mem_sol
    return sol, bound_type

def get_diff(name, real, custom):
    real = real.reshape(-1)
    custom = custom.reshape(-1)
    max_index = torch.argmax(torch.abs(real - custom)).item()
    real_val = real[max_index].item()
    custom_val = custom[max_index].item()
    print(name + " max diff", abs(real_val - custom_val), "index@", max_index, real_val, custom_val)

def perf(warm_up, itr, func):
    for ii in range(warm_up):
        func()
    torch.cuda.synchronize()
    a = datetime.now()
    for ii in range(itr):
        func()
    torch.cuda.synchronize()
    b = datetime.now()

    cost = (b - a).total_seconds() * 1000 / itr
    return cost

def test_helper(bsz, seqlen_q, seqlen_k, head_q, head_k, dim):
    q = torch.randn((bsz * seqlen_q, head_q, dim)).cuda().bfloat16()
    k = torch.randn((bsz * seqlen_k, head_k, dim)).cuda().bfloat16()
    v = torch.randn((bsz * seqlen_k, head_k, dim)).cuda().bfloat16()

    seqlens = torch.zeros([bsz]).cuda().int()
    seqlens[:] = seqlen_q
    cu_seqlens_q = F.pad(torch.cumsum(seqlens, dim = 0, dtype=torch.int32), (1, 0))
    seqlens[:] = seqlen_k
    cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim = 0, dtype=torch.int32), (1, 0))

    fwd_func_128x128x384 = partial(_flash_attn_fwd, q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True)
    fwd_func_128x64x384  = partial(_flash_attn_fwd, q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True, m_block_size=128, n_block_size=64, num_threads=384)
    fwd_func_64x64x256   = partial(_flash_attn_fwd, q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True, m_block_size=64, n_block_size=64, num_threads=256)
    out_128x128x384, lse_128x128x384 = fwd_func_128x128x384()
    out_128x64x384, lse_128x64x384 = fwd_func_128x64x384()
    out_64x64x256, lse_64x64x256 = fwd_func_64x64x256()

    ref_fwd_func = partial(_flash_attn_varlen_forward, q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, dropout_p=0.0, softmax_scale=dim**(-0.5), causal=True)

    ref_out, softmax_lsem, _, _ = ref_fwd_func()

    get_diff("fwd_128x128x384 ", out_128x128x384, ref_out)
    get_diff("fwd_128x64x384 ", out_128x64x384, ref_out)
    get_diff("fwd_64x64x256", out_64x64x256, ref_out)

    return perf(10, 10, fwd_func_128x128x384),  perf(10, 10, fwd_func_128x64x384), perf(10, 10, fwd_func_64x64x256)




bsz = 1
seqlen_q = 32768
seqlen_k = 32768
head_q = 32
head_k = 4
dim = 128
fwd_cost_128x128x384, fwd_cost_128x64x384, fwd_cost_64x64x256 = test_helper(bsz, seqlen_q, seqlen_k, head_q, head_k, dim)

sol, _ = sol_us(bsz, seqlen_q, seqlen_k, head_q, head_k, dim, True, is_bwd = False, local_window_size = -1)
mfu_128x128x384 = sol / (fwd_cost_128x128x384 * 1000)
mfu_128x64x384 = sol / (fwd_cost_128x64x384 * 1000)
mfu_64x64x256 = sol / (fwd_cost_64x64x256 * 1000)
print("fwd_cost_128x128x384 %.3f ms, mfu %.2f" % (fwd_cost_128x128x384, mfu_128x128x384))
print("fwd_cost_128x64x384 %.3f ms, mfu %.2f" % (fwd_cost_128x64x384, mfu_128x64x384))
print("fwd_cost_64x64x256 %.3f ms, mfu %.2f" % (fwd_cost_64x64x256, mfu_64x64x256))



import torch, math
import torch.nn.functional as F

from datetime import datetime
from functools import partial
from einops import rearrange, repeat
from flash_attn_interface import _flash_attn_forward, _flash_attn_backward

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

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


def mfu_us(b, s, h, d, d_v, is_bwd, is_causal, cost):
    bwd_computation_scale = 2.5
    computation = s*s*d*b*h + s*s*d_v*b*h
    computation = computation if is_causal else 2 * computation
    
    sol=computation/989/1000/1000

    final_flops = computation / (cost*1000*1000) if not is_bwd else bwd_computation_scale*computation / (cost*1000*1000)
    final_sol = sol/cost if not is_bwd else bwd_computation_scale*sol/cost
    
    return final_sol,final_flops


def test_helper(b, s, h, d, d_v, causal = False, check_diff = True, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(b, s, h, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, s, h, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, s, h, d_v, device=device, dtype=dtype, requires_grad=True)

    q1 = q.clone().detach().reshape(b*s, h, d)
    k1 = k.clone().detach().reshape(b*s, h, d)
    v1 = v.clone().detach().reshape(b*s, h, d_v)

    rand_seqlen_q = torch.zeros([b]).int()
    rand_seqlen_k = torch.zeros([b]).int()
    rand_seqlen_q[:] = s
    rand_seqlen_k[:] = s
    cu_seqlens_q = F.pad(torch.cumsum(rand_seqlen_q, dim = 0, dtype=torch.int32), (1, 0)).cuda()
    cu_seqlens_k = F.pad(torch.cumsum(rand_seqlen_k, dim = 0, dtype=torch.int32), (1, 0)).cuda()
    
    softmax_scale = (q1.shape[-1]) ** (-0.5)
    f3_fwd_func = partial(_flash_attn_forward,
        q1,#q,
        k1,#k,
        v1,#v,
        None, #k_new,
        None, #v_new,
        None, #qv,
        None, #out,
        cu_seqlens_q, #cu_seqlens_q,
        cu_seqlens_k, #cu_seqlens_k,
        None, #cu_seqlens_k_new,
        None, #seqused_q,
        None, #seqused_k,
        s, #max_seqlen_q,
        s, #max_seqlen_k,
        None, #page_table,
        None, #kv_batch_idx,
        None, #leftpad_k,
        None, #rotary_cos,
        None, #rotary_sin,
        None, #q_descale,
        None, #k_descale,
        None, #v_descale,
        softmax_scale,
        causal)
    out, lse, *rest = f3_fwd_func()
    fwd_cost = perf(10, 10, f3_fwd_func)

    grad_out = torch.randn_like(out)
    dq, dk, dv = torch.empty_like(q1), torch.empty_like(k1), torch.empty_like(v1)
    fa3_bwd_func = partial(_flash_attn_backward,
        grad_out, #dout
        q1, #q,
        k1, #k,
        v1, #v,
        out,
        lse,
        cu_seqlens_q, #cu_seqlens_q,
        cu_seqlens_k, #cu_seqlens_k,
        None, #sequed_q,
        None, #sequed_k,
        s, #max_seqlen_q,
        s, #max_seqlen_k,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        deterministic=False)
    bwd_cost = perf(10, 10, fa3_bwd_func)

    
    fwd_mfu, fwd_flops = mfu_us(b, s, h, d, d_v, False, causal, fwd_cost * 1000)
    bwd_mfu, bwd_flops = mfu_us(b, s, h, d, d_v, True, causal, bwd_cost * 1000)

    if check_diff:
        # ref
        ref_out = attention_ref(q, k, v, causal = causal)[0]
        ref_out.backward(grad_out.reshape(ref_out.shape), retain_graph = True)        
        get_diff("fwd ", ref_out, out)
        get_diff("dq ", q.grad, dq)
        get_diff("dk ", k.grad, dk)
        get_diff("dv ", v.grad, dv)
    
    print("(bshd)=(%d,%d,%d,%d) causal=%d, fwd = (lat=%.3f ms, sol=%.2f, tflops=%.1f), bwd = (lat=%.3f ms, sol%.2f, tflops=%.1f)" % (b, s, h, d, causal, fwd_cost,fwd_mfu, fwd_flops, bwd_cost,bwd_mfu, bwd_flops))

head_dim = 192
head_dimv = 128
#for bsz, seq in zip([8, 4, 1, 1, 1], [8*1024, 16*1024, 32*1024, 64*1024, 128*1024]):
for bsz, seq in zip([2], [8*1024]):
    test_helper(bsz, seq, 8, head_dim, head_dimv, causal=True, check_diff=False)
    
    
    
    
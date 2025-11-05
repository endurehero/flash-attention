import torch
from functools import partial
from interface import _flash_attn_fwd, _flash_attn_bwd
from datetime import datetime

def get_diff(name, real, custom, is_print=True):
    real = real.reshape(-1)
    custom = custom.reshape(-1)
    max_index = torch.argmax(torch.abs(real - custom)).item()
    real_val = real[max_index].item()
    custom_val = custom[max_index].item()
    
    max_diff = abs(real_val - custom_val)
    if is_print:
        print(name + " max diff", max_diff, "index@", max_index, real_val, custom_val)
    return max_diff

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

def get_mfu(costs:float, seqlen:torch.Tensor, num_heads:int, head_dim:int, is_bwd:bool=False):
    computation = 0
    memory_access = 0
    
    seq_num = seqlen.shape[0]
    for ii in range(seq_num):
        cur_org_seq = seqlen[ii].cpu().item()
        computation = computation + (cur_org_seq * cur_org_seq)
        memory_access = memory_access + cur_org_seq*2 + cur_org_seq*2
    
    computation=2*computation*num_heads*head_dim
    memory_access = (memory_access * num_heads * head_dim) * 2 # bf16

    if is_bwd:
        computation = computation * 2
    
    th_flops = get_gpu_flops()
    bandwidth = get_gpu_bandwidth()
    
    compute_cost_sol_ms = (computation / th_flops) * 1000
    memory_cost_sol_ms = (memory_access / bandwidth) * 1000

    final_cost_sol_ms = compute_cost_sol_ms if compute_cost_sol_ms > memory_cost_sol_ms else memory_cost_sol_ms
    
    return final_cost_sol_ms / costs

def attention_ref(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
) -> torch.Tensor:
    total_query_len, num_q_heads, head_dim = q.shape
    total_key_len, num_k_heads, _ = k.shape
    num_share_q_heads = num_q_heads // num_k_heads
    batch_size = cu_seqlens_q.shape[0] - 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    # get mask
    mask = torch.zeros(
        (total_query_len, total_key_len), dtype=torch.bool, device=q.device
    )
    for b in range(batch_size):
        q_len, k_len = (
            cu_seqlens_q[b + 1] - cu_seqlens_q[b],
            cu_seqlens_k[b + 1] - cu_seqlens_k[b],
        )
        k_max_ids = (
            torch.arange(k_len, device=q.device)
        )
        q_ids = torch.arange(q_len, device=q.device)
        mask[
            cu_seqlens_q[b] : cu_seqlens_q[b + 1], cu_seqlens_k[b] : cu_seqlens_k[b + 1]
        ] = (q_ids[:, None] >= k_max_ids[None, :])
    # attention
    qk = (
        torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(num_share_q_heads, 1))
        * sm_scale
    )
    qk = qk.masked_fill_(~mask[None, ...], -torch.inf)
    # query from beginning of the sequence can't attend to any compressed key
    qk = qk.softmax(dim=-1, dtype=torch.float32)
    qk = qk.nan_to_num(0)
    attn_output = torch.einsum(
        "hqk,khd->qhd", qk.to(v.dtype), v.repeat_interleave(num_share_q_heads, 1)
    )
    return attn_output

def test():
    torch.manual_seed(42)
    num_heads = 6
    num_heads_kv = 6
    head_dim = 128
    deterministic = False
    sm_margin = 0

    seqlens = torch.LongTensor([8192, 8192, 8192]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    max_seqlen = seqlens.max().item()
    
    batch_size = cu_seqlens.numel() - 1
    max_seqlen = seqlens[0].item()
    q = (
        torch.empty(cu_seqlens[-1], num_heads, head_dim, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    k = (
        torch.empty(cu_seqlens[-1], num_heads_kv, head_dim, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    v = (
        torch.empty(cu_seqlens[-1], num_heads_kv, head_dim, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )

    q_ref = q.clone().detach()
    k_ref = k.clone().detach()
    v_ref = v.clone().detach()
    q_ref.requires_grad_()
    k_ref.requires_grad_()
    v_ref.requires_grad_()
    q_ref.retain_grad()
    k_ref.retain_grad()
    v_ref.retain_grad()

    q = q.reshape(batch_size, max_seqlen, num_heads, head_dim)
    k = k.reshape(batch_size, max_seqlen, num_heads_kv, head_dim)
    v = v.reshape(batch_size, max_seqlen, num_heads_kv, head_dim)
    

    softmax_scale = head_dim**(-0.5)
    causal = True

    
    fwd_func = partial(_flash_attn_fwd, q=q, k=k, v=v, softmax_scale=softmax_scale, causal=causal, return_lse=True)
    out, lse = fwd_func()

    fwd_cost = perf(10, 10, fwd_func)
    fwd_mfu = 0 #get_mfu(fwd_cost, seqlens, num_heads, head_dim)
    print("fwd cost %.3fms, mfu %.2f" % (fwd_cost, fwd_mfu))

    dout = torch.randn_like(q)
    bwd_func = partial(_flash_attn_bwd, dout=dout, q=q, k=k, v=v, out=out, lse=lse,softmax_scale=softmax_scale, causal=causal)
    dq, dk, dv = bwd_func()

    if deterministic:
        dq_bak = dq.clone().detach()
        dk_bak = dk.clone().detach()
        dv_bak = dv.clone().detach()
        dq.zero_()
        dk.zero_()
        dv.zero_()
        bwd_func()
        get_diff("deterministic dq", dq_bak, dq)
        get_diff("deterministic dk", dk_bak, dk)
        get_diff("deterministic dv", dv_bak, dv)


    bwd_cost = perf(10, 10, bwd_func)

    bwd_mfu = 0 #get_mfu(bwd_cost, seqlens, num_heads, head_dim, True)
    print("bwd cost %.3fms, mfu %.2f"% (bwd_cost, bwd_mfu))

    
    out_ref = attention_ref(q_ref, k_ref, v_ref, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale)
    get_diff("fwd ", out_ref, out)
    out_ref.backward(dout)
    get_diff("dq ", q_ref.grad, dq)
    get_diff("dk ", k_ref.grad, dk)
    get_diff("dv ", v_ref.grad, dv)

test()




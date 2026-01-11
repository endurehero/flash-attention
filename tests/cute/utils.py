import torch
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
    elif "GB200" in name:
        return 8e12
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
    elif "GB200" in name:
        return 2500e12
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
        computation = computation * 2.5
    
    th_flops = get_gpu_flops()
    bandwidth = get_gpu_bandwidth()
    
    compute_cost_sol_ms = (computation / th_flops) * 1000
    memory_cost_sol_ms = (memory_access / bandwidth) * 1000

    final_cost_sol_ms = compute_cost_sol_ms if compute_cost_sol_ms > memory_cost_sol_ms else memory_cost_sol_ms
    
    return final_cost_sol_ms / costs
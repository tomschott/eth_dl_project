import torch
import json

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            if obj.numel() > 1: 
                return obj.tolist()
            if obj.numel() == 1: 
                return obj.item()
            return None
        return json.JSONEncoder.default(self, obj)

def flatten_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.reshape(-1))
    return torch.cat(grads)

def flatten_params(model):
    params = [p.detach().reshape(-1) for p in model.parameters()]
    return torch.cat(params)

def flatten_eigvec_structure(ev, dtype=None, device=None):
    if isinstance(ev, torch.Tensor):
        v = ev.reshape(-1)
    else:
        # Handle nested lists/tuples
        chunks = []
        def _collect(x):
            if isinstance(x, torch.Tensor): 
                chunks.append(x.reshape(-1))
            elif isinstance(x, (list, tuple)): 
                for y in x:
                    _collect(y)
            else: 
                chunks.append(torch.as_tensor(x).reshape(-1))
        _collect(ev)
        v = torch.cat(chunks)
        
    if dtype: 
        v = v.to(dtype=dtype)
    if device: 
        v = v.to(device)
    return v

def layer_spectral_stats(S, k_top=5, eps=1e-12):
    s_max = S[0].item()
    s_nonzero = S[S > eps]
    cond = (s_nonzero.max() / s_nonzero.min()).item() if s_nonzero.numel() > 0 else float('nan')
    frob = torch.sqrt((S ** 2).sum()).item()
    
    S_sum = S.sum()
    eff_rank = 0.0
    if S_sum.item() > eps:
        p = S / S_sum
        entropy = -(p * (p + eps).log()).sum()
        eff_rank = torch.exp(entropy).item()

    k = min(k_top, S.shape[0])
    topk_ratio = ((S[:k] ** 2).sum() / ((S ** 2).sum() + eps)).item()

    return s_max, cond, frob, eff_rank, topk_ratio
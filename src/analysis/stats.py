import torch
from pyhessian import hessian
from src.utils import metrics as utils

def _flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])

def _unflatten_vector(vec, like_tensors):
    chunks = []
    offset = 0
    for t in like_tensors:
        numel = t.numel()
        chunk = vec[offset:offset + numel].view_as(t)
        chunks.append(chunk)
        offset += numel
    return chunks

def _approx_hessian_top_eig(net, criterion, input_img, label, max_iters=10, eps=1e-12):
    params = [p for p in net.parameters() if p.requires_grad]
    loss = criterion(net(input_img), label)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    v = _flatten_tensors([torch.randn_like(g) for g in grads])
    v = v / (v.norm() + eps)
    eigval = torch.tensor(0.0, device=v.device)

    for _ in range(max_iters):
        v_list = _unflatten_vector(v, grads)
        hvp = torch.autograd.grad(grads, params, grad_outputs=v_list, retain_graph=True)
        hvp_flat = _flatten_tensors(hvp)
        eigval = torch.dot(v, hvp_flat) / (torch.dot(v, v) + eps)
        v = hvp_flat / (hvp_flat.norm() + eps)

    return eigval.item(), v


def spectral_entropy_from_singular_values(S, eps=1e-12, normalized=True):
    """
    S: 1D tensor of singular values (nonnegative).
    Returns entropy (optionally normalized to [0,1]).
    """
    e = S.pow(2)
    p = e / (e.sum() + eps)
    H = -(p * (p + eps).log()).sum()  # natural log

    if normalized:
        r = S.numel()
        if r > 1:
            H = H / (torch.log(torch.tensor(r, device=S.device, dtype=S.dtype)) + eps)
        else:
            H = torch.zeros_like(H)
    return H


def compute_statistics(results, net, criterion, input_img, label,
                       num_batch, prev_params, flat_grad, optimizer_key,
                       k_top=10):
    """
    Computes SVD of weights, Hessian stats, and update geometry.
    """
    device = flat_grad.device
    
    # 1. SVDs of weight matrices
    linear_layers = [
        (name, layer) for name, layer in net.named_modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("No nn.Linear layers found for spectral stats.")

    with torch.no_grad():
        for i, (_, layer) in enumerate(linear_layers, 1):
            U, S, Vh = torch.linalg.svd(layer.weight, full_matrices=False)
            results[optimizer_key][f'U{i}_list'].append(U)
            results[optimizer_key][f'V{i}_list'].append(Vh)
            
            stats = utils.layer_spectral_stats(S, k_top=k_top)
            (s_max, cond, frob, eff_rank, topk) = stats
            
            results[optimizer_key][f'L{i}_top_eigenvalues'].append(s_max)
            results[optimizer_key][f'L_{i}condition_number'].append(cond)
            results[optimizer_key][f'L{i}_frobenius_norm'].append(frob)
            results[optimizer_key][f'L{i}_effective_rank'].append(eff_rank)
            results[optimizer_key][f'L{i}_topk_energy_ratio'].append(topk)

            # --- entropy ---
            H = spectral_entropy_from_singular_values(S, normalized=True)
            results[optimizer_key].setdefault(f'L{i}_spectral_entropy', []).append(H.item())

    # 2. Hessian + Alignment
    # Re-wrap data for PyHessian
    use_cuda = input_img.is_cuda
    if use_cuda:
        hessian_comp = hessian(
            net, criterion, data=(input_img, label), cuda=True
        )
        
        # Get top 2 eigenvalues/vectors
        top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=2)
        results[optimizer_key]['sharpness'].append(top_eigenvalues[-2]) # Second largest

        # Flatten top eigenvector
        ev = top_eigenvectors[-1] if isinstance(top_eigenvectors, list) else top_eigenvectors
        v_top = utils.flatten_eigvec_structure(ev, dtype=flat_grad.dtype, device=device)
    else:
        # This is only meant as a fallback for CPU mode
        eigval, v_top = _approx_hessian_top_eig(net, criterion, input_img, label)
        results[optimizer_key]['sharpness'].append(eigval)

    # Cosine(Gradient, Hessian Top Eigenvector)
    cos_g_v = torch.dot(flat_grad, v_top) / (flat_grad.norm() * v_top.norm() + 1e-12)
    results[optimizer_key]['grad_hessian_cos'].append(cos_g_v.item())

    # 3. Update Geometry
    flat_params = utils.flatten_params(net)
    if prev_params is not None:
        delta_theta = flat_params - prev_params
        cos_update_grad = torch.dot(delta_theta, -flat_grad) / (
            delta_theta.norm() * flat_grad.norm() + 1e-12
        )
        results[optimizer_key]['update_norm'].append(delta_theta.norm().item())
        results[optimizer_key]['grad_norm'].append(flat_grad.norm().item())
        results[optimizer_key]['update_grad_cos'].append(cos_update_grad.item())

    return flat_params.detach().clone(), results

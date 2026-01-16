import argparse
import random
import torch
import torch.nn as nn
import json
import os
from src.models.mlp import ConfigurableMLP
from src.dataloader.dataloaders import get_dataloaders
from src.utils.metrics import evaluate, TensorEncoder
from src.training.engine import train_one_epoch

def _parse_args():
    parser = argparse.ArgumentParser(description="Train models and optionally collect stats.")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=15)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--stats-enabled", action="store_true")
    parser.add_argument("--optimizers", default="Muon,AdamW")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def _get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )

def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def _count_linear_layers(model):
    return sum(1 for layer in model.modules() if isinstance(layer, nn.Linear))

def _build_metrics(num_layers):
    metrics = [
        'sharpness', 'grad_hessian_cos', 'update_norm',
        'grad_norm', 'update_grad_cos',
    ]
    for i in range(1, num_layers + 1):
        metrics.extend([
            f'U{i}_list', f'V{i}_list',
            f'L{i}_top_eigenvalues', f'L_{i}condition_number',
            f'L{i}_frobenius_norm', f'L{i}_effective_rank',
            f'L{i}_topk_energy_ratio',
        ])
    return metrics

def _get_optim_config_path(num_hidden_layers):
    return f"results/optim_configs_h{num_hidden_layers}.json"

def _is_scalar_metric(metric_name):
    if metric_name.startswith("U") or metric_name.startswith("V"):
        return False
    return True

def _average_results(run_results):
    if not run_results:
        return {}
    avg_results = {}
    optimizers = run_results[0].keys()
    for opt in optimizers:
        avg_results[opt] = {}
        metrics = run_results[0][opt].keys()
        for metric in metrics:
            if not _is_scalar_metric(metric):
                continue
            per_run_series = [r[opt][metric] for r in run_results]
            min_len = min(len(series) for series in per_run_series)
            avg_series = []
            for i in range(min_len):
                vals = [series[i] for series in per_run_series]
                avg_series.append(sum(vals) / len(vals))
            avg_results[opt][metric] = avg_series
    return avg_results

def main():
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    optimizer_names = [opt.strip() for opt in args.optimizers.split(",") if opt.strip()]
    print(f"Using device: {device}")
    train_loader, val_loader = get_dataloaders(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    
    # Load Configs
    config_path = _get_optim_config_path(args.num_hidden_layers)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            optim_configs = json.load(f)
    elif os.path.exists('results/optim_configs.json'):
        with open('results/optim_configs.json', 'r') as f:
            optim_configs = json.load(f)
    else:
        optim_configs = {'Muon': {'lr': 0.01}, 'AdamW': {'lr': 0.01}}

    probe_model = ConfigurableMLP(
        num_hidden_layers=args.num_hidden_layers,
        hidden_dim=args.hidden_dim,
    )
    num_layers = _count_linear_layers(probe_model)
    metrics = _build_metrics(num_layers)
    run_results = []

    # --- Run Experiments ---
    for run_idx in range(1, args.num_runs + 1):
        results = {opt: {m: [] for m in metrics} for opt in optimizer_names}
        print(f"\n=== Starting Run {run_idx}/{args.num_runs} ===")

        for opt_name in optimizer_names:
            print(f"\n=== Starting Experiment: {opt_name} ===")
            
            # 1. Setup Model & Optimizer
            model = ConfigurableMLP(
                num_hidden_layers=args.num_hidden_layers,
                hidden_dim=args.hidden_dim,
            ).to(device)
            
            if opt_name == 'Muon':
                optimizer = torch.optim.Muon(model.parameters(), lr=optim_configs['Muon']['lr'])
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=optim_configs['AdamW']['lr'])

            prev_params = None 
            
            # 2. Training Loop
            for epoch in range(args.num_epochs):
                loss, acc, prev_params, results = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    optimizer_name=opt_name,
                    results=results,
                    prev_params=prev_params,
                    log_interval=args.log_every_n_steps,
                    compute_stats=args.stats_enabled
                )
                
                val_acc = evaluate(model, device, val_loader)
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.2f}")

            # 3. Save Model
            save_path = f'results/models/{opt_name.lower()}_model_fashion_h{args.num_hidden_layers}_v{run_idx}.pt'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved {opt_name} model to {save_path}")

        run_results.append(results)

    # --- Save Statistics ---
    if args.stats_enabled:
        os.makedirs('results', exist_ok=True)
        stats_path = f"results/training_stats_h{args.num_hidden_layers}.json"
        with open(stats_path, 'w') as f:
            json.dump(run_results, f, indent=4, cls=TensorEncoder)
        print(f"Saved training stats to {stats_path}.")
        if args.num_runs > 1:
            avg_results = _average_results(run_results)
            avg_path = f"results/training_stats_avg_h{args.num_hidden_layers}.json"
            with open(avg_path, 'w') as f:
                json.dump(avg_results, f, indent=4, cls=TensorEncoder)
            print(f"Saved averaged training stats to {avg_path}.")

if __name__ == "__main__":
    main()

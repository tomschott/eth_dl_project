import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from src.models.mlp import ConfigurableMLP
from src.dataloader.dataloaders import get_dataloaders
from src.training.engine import train_one_epoch

def _parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for optimizers.")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=15)
    parser.add_argument("--optimizers", default="AdamW,Muon")
    parser.add_argument("--learning-rates", default="1e-4,5e-4,1e-3,5e-3,1e-2,5e-2")
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

def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]

def _parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]

def get_optimizer(name, model, lr):
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr)
    elif name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif name == "Muon":
        return optim.Muon(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def evaluate_loss(model, device, loader):
    """
    Specific evaluation for tuning that returns average loss.
    (utils.evaluate only returns accuracy)
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_fn(output, target).item()
    
    return total_loss / len(loader)

def main():
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    optimizers_to_test = _parse_list(args.optimizers)
    learning_rates = _parse_float_list(args.learning_rates)
    print(f"Using device: {device}")
    
    # Use the shared data loader to ensure consistency with training/quantization
    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    # ==========================================
    # Phase 1: Hyperparameter Sweep
    # ==========================================
    best_configs = {} 
    criterion = nn.CrossEntropyLoss()

    print("--- Phase 1: Hyperparameter Sweep (Finding Best LR) ---")
    
    for opt_name in optimizers_to_test:
        best_loss = float('inf')
        best_lr = 0.0
        
        for lr in learning_rates:
            model = ConfigurableMLP(
                num_hidden_layers=args.num_hidden_layers,
                hidden_dim=args.hidden_dim,
            ).to(device)
            optimizer = get_optimizer(opt_name, model, lr)
            
            # Track minimum validation loss found during this training run
            run_min_val_loss = float('inf')
            
            for epoch in range(args.search_epochs):
                # Use shared engine to train one epoch
                # We don't need the stats here, so compute_stats=False
                train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    optimizer_name=opt_name,
                    results=None,       # No tracking needed
                    prev_params=None,
                    compute_stats=False
                )
                
                # Val Check
                val_loss = evaluate_loss(model, device, test_loader)
                run_min_val_loss = min(val_loss, run_min_val_loss)
            
            print(f"[{opt_name}] LR={lr} | Min Val Loss={run_min_val_loss:.4f}")
            
            if run_min_val_loss < best_loss:
                best_loss = run_min_val_loss
                best_lr = lr
                
        best_configs[opt_name] = {'lr': best_lr, 'min_loss': best_loss}
        print(f"--> Best {opt_name}: LR={best_lr}, Loss={best_loss:.4f}\n")

    # ==========================================
    # Phase 2: Determine Common Loss (CL)
    # ==========================================
    all_min_losses = [cfg['min_loss'] for cfg in best_configs.values()]
    COMMON_LOSS = max(all_min_losses) 
    COMMON_LOSS_TARGET = COMMON_LOSS + 0.005 

    print("--- Phase 2: Calculating Common Loss (CL) ---")
    #print(f"Best losses: {[f'{k}: {v['min_loss']:.4f}' for k,v in best_configs.items()]}")
    print(f"Common Loss Target (CL): {COMMON_LOSS_TARGET:.4f}")
    
    best_configs['CL_target'] = COMMON_LOSS_TARGET

    os.makedirs('results', exist_ok=True)
    config_path = f"results/optim_configs_h{args.num_hidden_layers}.json"
    with open(config_path, 'w') as f:
        json.dump(best_configs, f, indent=4)
    print(f"Saved configs to {config_path}")

if __name__ == "__main__":
    main()

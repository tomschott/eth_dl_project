import argparse
import random
import torch
import copy
import numpy as np
import csv
import os
from src.models.mlp import ConfigurableMLP
from src.dataloader.dataloaders import get_dataloaders
from src.utils import metrics as utils

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate quantized model accuracy.")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--bits", default="8,4,2")
    parser.add_argument("--adam-pattern", default="results/models/adamw_model_fashion_h3_v{}.pt")
    parser.add_argument("--muon-pattern", default="results/models/muon_model_fashion_h3_v{}.pt")
    parser.add_argument("--num-runs", type=int, default=6, help="Number of versions (v1 to vN)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default="results/quantization_results.csv", help="Path to save results CSV")
    return parser.parse_args()

def _parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass

def fake_quantize_weight(weight, bits, percentile=0.999):
    """
    Mathematically simulates symmetric quantization for N bits.

    """
    if weight.abs().max() == 0:
        return weight
    
    max_int = 2**(bits - 1) - 1
    
    # 1. Calculate Scale
    abs_weight = weight.abs().reshape(-1)
    scale = torch.quantile(abs_weight, percentile) / max_int 
    
    # 2. Quantize (Divide -> Round -> Clamp)
    weight_int = (weight / scale).round().clamp(-max_int, max_int)
    
    # 3. De-quantize (Multiply back by scale)
    weight_sim = weight_int * scale
    return weight_sim

def apply_fake_quant(model, bits):
    """Iterates through all Linear layers and destructively quantizes weights."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.data = fake_quantize_weight(module.weight.data, bits)
    return model

def _linear_layers(model):
    return [(name, layer) for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]

def _quant_diffs(model, bits, eps=1e-12):
    """
    Returns per-layer quantization diagnostics for Linear weights:
      - abs_l2: ||W - Wq||
      - rel_l2: ||W - Wq|| / ||W||
      - w_norm: ||W||
      - max_abs: max(|W|)
      - scale: max(|W|)/max_int used by quantizer
    """
    out = {}
    max_int = 2**(bits - 1) - 1

    with torch.no_grad():
        for name, layer in _linear_layers(model):
            W = layer.weight.data
            Wq = fake_quantize_weight(W, bits)

            diff = W - Wq
            abs_l2 = diff.norm().item()
            w_norm = (W.norm().item() + eps)
            rel_l2 = abs_l2 / w_norm

            max_abs = W.abs().max().item()
            scale = (max_abs / max_int) if max_abs > 0 else 0.0

            out[name] = {
                "abs_l2": abs_l2,
                "rel_l2": rel_l2,
                "w_norm": w_norm,
                "max_abs": max_abs,
                "scale": scale,
            }
    return out

def save_to_csv(results, output_path):
    """
    Saves the raw results to a CSV file in long format.
    Format: Model, Quantization, Version, Accuracy
    """
    # Ensure directory exists
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    headers = ["Model", "Quantization", "Version", "Accuracy"]
    
    try:
        with open(output_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for model_name, bit_configs in results.items():
                for bit_label, accuracies in bit_configs.items():
                    for i, acc in enumerate(accuracies):
                        # Version is index + 1 (e.g., v1, v2)
                        writer.writerow([model_name, bit_label, i + 1, acc])
        
        print(f"\n[Success] Detailed results saved to: {output_path}")
    except IOError as e:
        print(f"\n[Error] Could not save CSV: {e}")

def main():
    args = _parse_args()
    _set_seed(args.seed)
    quant_bits = _parse_int_list(args.bits)
    _, test_loader = get_dataloaders(args.batch_size)
    
    model_groups = {
        "AdamW": args.adam_pattern,
        "Muon":  args.muon_pattern
    }

    # Storage: results[model_name][bit_width] = [acc_v1, acc_v2, ...]
    results = {name: {'FP32': []} for name in model_groups}
    for name in results:
        for b in quant_bits:
            results[name][f"INT{b}"] = []

    print(f"Evaluating {args.num_runs} versions per model...")
    
    for name, pattern in model_groups.items():
        print(f"Processing {name} models...")
        
        for i in range(1, args.num_runs + 1):
            file_path = pattern.format(i)
            
            # Load Baseline
            model = ConfigurableMLP().to(args.device)
            try:
                model.load_state_dict(torch.load(file_path, map_location=args.device))
            except FileNotFoundError:
                print(f"  [Warning] File not found: {file_path}, skipping.")
                # We append None or 0.0 to keep indices aligned, or handle skipping logic
                # For simplicity, we assume files exist or we just skip appending
                continue

            # Evaluate FP32
            acc_fp32 = utils.evaluate(model, args.device, test_loader)
            results[name]['FP32'].append(acc_fp32)
            
            # Evaluate Quantized
            for bits in quant_bits:
                model_q = copy.deepcopy(model)
                apply_fake_quant(model_q, bits=bits)
                acc_q = utils.evaluate(model_q, args.device, test_loader)
                results[name][f"INT{bits}"].append(acc_q)
            
            # Print progress for this specific version
            print(f"  v{i}: FP32={acc_fp32:.2f}% | " + " | ".join(
                [f"INT{b}={results[name][f'INT{b}'][-1]:.2f}%" for b in quant_bits]
            ))

    # --- PRINT SUMMARY TABLE ---
    model_col = 15
    metric_col = 18 
    
    header_keys = ["FP32"] + [f"INT{b}" for b in quant_bits]
    header = f"\n{'Model (Avg)':<{model_col}} | " + " | ".join(
        f"{col:<{metric_col}}" for col in header_keys
    )
    
    print("\n" + "=" * len(header.strip()))
    print("AGGREGATE RESULTS (Mean ± Std)")
    print("=" * len(header.strip()))
    print(header)
    print("-" * len(header.strip()))

    for name in model_groups:
        row = f"{name:<{model_col}} | "
        
        for key in header_keys:
            data = results[name].get(key, [])
            if len(data) > 0:
                mean = np.mean(data)
                std = np.std(data)
                row += f"{mean:>6.2f} ± {std:<5.2f} | "
            else:
                row += f"{'N/A':<{metric_col}} | "
        
        print(row.strip(" | "))

    print("=" * len(header.strip()))

    # --- SAVE TO CSV ---
    save_to_csv(results, args.output_file)

if __name__ == "__main__":
    main()
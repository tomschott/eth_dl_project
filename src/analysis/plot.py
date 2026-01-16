import torch
import matplotlib.pyplot as plt
from src.models.mlp import ThreeLayerMLP

# 1. Load your models
model_adam = ThreeLayerMLP()
model_adam.load_state_dict(torch.load("results/models/adamw_model_fashion.pt", map_location="cpu"))

model_muon = ThreeLayerMLP()
model_muon.load_state_dict(torch.load("results/models/muon_model_fashion.pt", map_location="cpu"))

def get_singular_values(model, layer_name="l1"):
    # Extract weight matrix
    # Note: Linear layer weights are shape (out_features, in_features)
    W = getattr(model, layer_name).weight.data
    
    # Compute SVD
    # S is the vector of singular values (sigma)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Normalize for fair comparison (so max value is 1.0)
    # This helps compare "shape" rather than just "magnitude"
    S_norm = S / S.max()
    return S, S_norm

# Get values for Layer 1 (The 784 -> 15 bottleneck)
# This is usually the most interesting layer to visualize
S_adam, S_adam_norm = get_singular_values(model_adam, "l1")
S_muon, S_muon_norm = get_singular_values(model_muon, "l1")

# --- PLOTTING ---
plt.figure(figsize=(10, 5))

# Plot 1: Raw Values (To see magnitude differences)
plt.subplot(1, 2, 1)
plt.plot(S_adam.numpy(), 'r-o', label='AdamW')
plt.plot(S_muon.numpy(), 'b-o', label='Muon')
plt.title("Raw Singular Values (Layer 1)")
plt.xlabel("Singular Value Index")
plt.ylabel("Value ($\sigma$)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Normalized (To see distribution shape/spikiness)
plt.subplot(1, 2, 2)
plt.plot(S_adam_norm.numpy(), 'r-o', label='AdamW (Normalized)')
plt.plot(S_muon_norm.numpy(), 'b-o', label='Muon (Normalized)')
plt.title("Normalized Spectrum (Shape)")
plt.xlabel("Singular Value Index")
plt.ylabel("Normalized Value ($\sigma / \sigma_{max}$)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("spectrum_comparison_l1.png")

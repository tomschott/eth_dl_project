# Optimizer Analysis & Quantization Experiment (Muon vs. AdamW)

This project compares the training dynamics of the **Muon** and **AdamW** optimizers on a simple MLP trained on **FashionMNIST**. It tracks advanced geometric metrics (like Hessian sharpness and spectral entropy) during training and measures how well the resulting models survive post-training quantization (**INT8, INT4, INT2**).

## Project Structure

The code is organized under `src/` to separate data loading, training logic, and analysis tools. The root directory contains the main execution script.

| Path                                 | Description                                                                                              |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------- |
| **`run_all.sh`** | **Entry Point:** A shell script to orchestrate the full pipeline (Fine-tune $\to$ Train $\to$ Quantize). |
| **`src/training/fine_tune.py`** | **Phase 1:** Sweeps learning rates to find the best configuration and calculates a "Common Loss" target. |
| **`src/training/train.py`** | **Phase 2:** Trains models using the best configs, tracks Hessian/SVD stats, and saves checkpoints.      |
| **`src/quantization/quantize.py`** | **Phase 3:** Applies fake quantization (INT8/4/2) to trained models and evaluates accuracy degradation.  |
| **`src/dataloader/dataloaders.py`** | Handles loading **FashionMNIST**. Does not use normalization to test raw distribution robustness.        |
| **`src/models/mlp.py`** | Defines the `ConfigurableMLP` architecture used in experiments.                                          |
| **`src/training/engine.py`** | Contains the shared `train_one_epoch` loop used by both fine-tuning and training phases.                 |
| **`src/analysis/stats.py`** | Computes heavy statistics (Hessian eigenvalues, spectral norms, entropy).                                |
| **`src/analysis/plot.py`** | Utilities for plotting singular value spectrum comparisons.                                              |
| **`src/utils/metrics.py`** | Helper functions for evaluation and JSON serialization.                                                  |

## Usage Guide

### 1. Install Dependencies

Ensure you have PyTorch installed. This project also relies on `pyhessian` for curvature analysis and `matplotlib` for plotting.

```bash
pip install torch torchvision tqdm pyhessian matplotlib numpy
```

### 2. Run the Full Experiment

1. **Fine-tune**: Finds optimal learning rates.
2. **Train**: Trains multiple runs of AdamW and Muon models with those rates.
3. **Quantize**: Evaluates the trained models at reduced precision.

Run everything (Fine-tune, Train, Quantize)
```bash
./run_all.sh --all
```
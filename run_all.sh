#!/usr/bin/env bash
set -euo pipefail

# Shared hyperparameters
DEVICE="mps"                     # cpu | cuda | mps
BATCH_SIZE=64
NUM_EPOCHS=20
NUM_HIDDEN_LAYERS=3
OPTIMIZERS="AdamW,Muon"

# Fine-tuning sweep params
FINE_TUNE_LEARNING_RATES="1e-4,5e-4,1e-3,5e-3,1e-2,5e-2"

# Training run params
NUM_RUNS=10
TRAIN_LOG_EVERY_N_STEPS=50
TRAIN_STATS_ENABLED="true"

# Quantization eval params
QUANT_BATCH_SIZE=1000
QUANT_BITS="8,4,2"
ADAM_PATTERN="results/models/adamw_model_fashion_h${NUM_HIDDEN_LAYERS}_v{}.pt"
MUON_PATTERN="results/models/muon_model_fashion_h${NUM_HIDDEN_LAYERS}_v{}.pt"
QUANT_OUTPUT="results/quantization_results.csv"

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [options]

Options:
  -h, --help        Show this help message
  --all             Run fine_tune, train, and quantize (default)
  --fine-tune       Run only fine_tune
  --train           Run only train
  --quantize        Run only quantize
EOF
}

RUN_FINE_TUNE=false
RUN_TRAIN=false
RUN_QUANTIZE=false

if [[ "$#" -eq 0 ]]; then
  RUN_FINE_TUNE=true
  RUN_TRAIN=true
  RUN_QUANTIZE=true
else
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --all)
        RUN_FINE_TUNE=true
        RUN_TRAIN=true
        RUN_QUANTIZE=true
        ;;
      --fine-tune)
        RUN_FINE_TUNE=true
        ;;
      --train)
        RUN_TRAIN=true
        ;;
      --quantize)
        RUN_QUANTIZE=true
        ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
fi

if [[ "${RUN_FINE_TUNE}" == "true" ]]; then
  echo "Running fine-tune sweep..."
  python -m src.training.fine_tune \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --search-epochs "${NUM_EPOCHS}" \
    --num-hidden-layers "${NUM_HIDDEN_LAYERS}" \
    --optimizers "${OPTIMIZERS}" \
    --learning-rates "${FINE_TUNE_LEARNING_RATES}"
fi

if [[ "${RUN_TRAIN}" == "true" ]]; then
  echo "Running training..."
  train_args=(
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --num-epochs "${NUM_EPOCHS}"
    --num-runs "${NUM_RUNS}"
    --num-hidden-layers "${NUM_HIDDEN_LAYERS}"
    --log-every-n-steps "${TRAIN_LOG_EVERY_N_STEPS}"
    --optimizers "${OPTIMIZERS}"
  )
  if [[ "${TRAIN_STATS_ENABLED}" == "true" ]]; then
    train_args+=(--stats-enabled)
  fi
  python -m src.training.train "${train_args[@]}"
fi

if [[ "${RUN_QUANTIZE}" == "true" ]]; then
  echo "Running quantization evaluation..."
  python -m src.quantization.quantize \
    --device "${DEVICE}" \
    --batch-size "${QUANT_BATCH_SIZE}" \
    --bits "${QUANT_BITS}" \
    --adam-pattern "${ADAM_PATTERN}" \
    --muon-pattern "${MUON_PATTERN}" \
    --num-runs "${NUM_RUNS}" \
    --output-file "${QUANT_OUTPUT}"
fi
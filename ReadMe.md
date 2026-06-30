# AdaptiveBelief — Memory Framework for Long-Horizon AI Agents

> **Model:** Qwen2.5-3B-Instruct | **Dataset:** LoCoMo | **Hardware:** RTX 3060 (12 GB VRAM)

AdaptiveBelief is a memory framework that tightly couples an **implicit belief state** with an **uncertainty-gated explicit decision trigger**, mediated by a frozen reasoner. It learns *when* to externalize structured memory, keeping inference bounded and interpretable while handling sparse evidence across arbitrarily long conversation trajectories.

---

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiments](#experiments)
- [Metrics & Expected Results](#metrics--expected-results)
- [Quick Reference](#quick-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

Long-horizon AI agents must compress and recall massive conversation histories. Existing memory designs are either:
- **explicit** (interpretable but brittle), or
- **implicit** (latent and stable but opaque).

**AdaptiveBelief** decouples these roles:

| Component | Role | Memory Type | Trainable |
|-----------|------|-------------|-----------|
| **Builder** ($f_\theta$) | Trajectory → Belief state | Implicit (latent) | Yes |
| **Uncertainty Estimator** ($g_\phi$) | Monitors belief reliability | — | Yes |
| **Retriever** | Belief → Decision trace | Explicit (symbolic) | Yes |
| **Hard Top-K + STE** | Gradient-sparse selection | — | Yes (via STE) |
| **Frozen Reasoner** | Decision trace → Answer | — | No |

The Builder uses a LoRA-adapted transformer to construct a compressed belief state $b_t \in \mathbb{R}^{768}$. A lightweight MLP estimates uncertainty $u_t = g_\phi(b_t)$. When $u_t \geq \tau$ (learned threshold), the Retriever externalizes a structured **decision trace** using **Hard Top-K with Straight-Through Estimator (STE)** for sparse, interpretable writes. The frozen reasoner (Qwen2.5-3B-Instruct, 4-bit quantized) receives only the trace at decision points, never the raw history.

Training uses a composite loss with uncertainty calibration, sparsity penalty, compression loss, and phase‑based threshold annealing.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TRAJECTORY                         │
│              (19 conversation sessions with timestamps)          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BUILDER (Trainable)                         │
│          f_θ: [b_{t-1}, o_t] → b_t (latent belief state)        │
│                    dimension: d = 768 (Qwen3B alignment)         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ BELIEF STATE  │
                      │   b_t ∈ R^d   │
                      └───────┬───────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ UNCERTAINTY │   │ COMPRESSION │   │  VERSIONING │
    │  Estimator  │   │   Loss      │   │  (optional) │
    │ g_φ(b_t)→u_t│   │ L_recon     │   │             │
    └──────┬──────┘   └─────────────┘   └─────────────┘
           │
           ▼
    u_t ≥ τ? ───NO───► Continue (no explicit write)
           │
          YES
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVER (Trainable)                       │
│         Hard Top-K + STE selection over belief positions         │
│              → Structured Decision Trace (symbolic)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FROZEN REASONER (Qwen2.5-3B-Instruct, frozen)    │
│    Input: Decision trace (context) + Question + Choices          │
│    Output: Answer prediction                                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │    OUTPUT     │
                      │   Answer +    │
                      │  Uncertainty  │
                      └───────────────┘
```

---

## Installation



```angular2html
# Clone and enter project
git clone <repo-url> AdaptiveBelief
cd AdaptiveBelief

# Create virtual environment (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install quantization support
pip install bitsandbytes accelerate

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('VRAM:', torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0, 'GB')"
```

**requirements.txt**:
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
wandb>=0.15.0
pyyaml>=6.0
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
```

---

## Project Structure

All source files are fully implemented. The directory layout:

```
AdaptiveBelief/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── default.yaml
│   └── rtx3060_3b.yaml
│
├── data/
│   ├── raw/
│   │   └── locomo10.json
│   ├── processed/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── preprocess.py
│
├── src/
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── builder_3b.py          # f_θ: LoRA-adapted belief state updater
│   │   ├── uncertainty.py         # g_φ: 2-layer MLP uncertainty estimator
│   │   ├── retriever.py           # Hard Top-K + STE, outputs decision trace JSON
│   │   └── belief_state.py        # b_t initialization, update, versioning
│   │
│   ├── reasoner/
│   │   ├── __init__.py
│   │   ├── frozen_llm_3b.py       # Qwen2.5-3B-Instruct 4-bit wrapper, fully frozen
│   │   ├── prompt_builder.py      # Decision trace → inference prompt
│   │   └── output_parser.py       # Parse single-letter answer from LLM output
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_3b.py            # Main training loop with threshold annealing
│   │   ├── train_3b_optimized.py  # Optimized training (faster)
│   │   ├── train_3b_optimized_full.py  # Full optimized training with LoRA
│   │   ├── train_3b_fixed.py      # Fixed training with proper causal LM
│   │   ├── train_3b_resume.py     # Resume training from checkpoint
│   │   ├── losses.py              # L_task, L_uncertainty, L_sparsity, L_compression
│   │   ├── optimizer.py           # AdamW with phase-based LR schedule
│   │   └── data_loader.py         # LoCoMo batch loader
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Accuracy, ECE, latency runner
│   │   ├── metrics.py             # All metric calculations
│   │   └── ablation.py            # Runs all 7 configurations and saves results
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # WandB + TensorBoard logging
│       ├── checkpoint.py          # Save / load model weights
│       └── ste.py                 # Straight-through estimator
│
├── scripts/
│   ├── run_experiment.py          # CLI entry point for any single experiment
│   ├── evaluate_all.py            # Runs all baselines and ablations, saves CSV
│   ├── final_evaluation.py        # Comprehensive final evaluation
│   ├── final_evaluation_with_lora.py        # Comprehensive final evaluation with LoRA attach
│   ├── test_setup.py              # Test GPU and model setup
│   ├── test_trained_model.py      # Test trained LoRA model
│   ├── minimal_inference.py      # Sample Inference Check
│   ├── simple_eval.py            # 
│   ├── convert_checkpoint.py     # best checkpoint
│   └── validate_data.py          # Data Validate
│ 
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── traces/
│   ├── results/
│   └── best_lora_model/           # Saved LoRA weights after training
│
└── tests/
    ├── test_builder.py
    ├── test_uncertainty.py
    ├── test_retriever.py
    ├── test_hard_topk.py
    └── test_integration.py
```

---

## Configuration

The main training config for RTX 3060 is `config/rtx3060_3b.yaml`:

```yaml
base_model: "Qwen/Qwen2.5-3B-Instruct"
model_type: "causal_lm"
belief_dim: 768
hidden_dim: 256

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

training:
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 100
  weight_decay: 0.01
  max_seq_length: 2048
  max_sessions: 10
  save_steps: 500
  eval_steps: 100
  logging_steps: 10
  max_grad_norm: 1.0

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

loss_weights:
  task: 1.0
  uncertainty: 0.5
  sparsity: 0.1
  compression: 0.2

threshold:
  initial: 0.3
  final: 0.7
  anneal_steps: 500

top_k:
  dynamic: true
  base_k: 3

data:
  train_file: "data/processed/train.json"
  val_file: "data/processed/val.json"
  test_file: "data/processed/test.json"

output:
  checkpoint_dir: "outputs/checkpoints"
  log_dir: "outputs/logs"
  result_dir: "outputs/results"
  trace_dir: "outputs/traces"
```

Loss coefficients and threshold annealing schedule:

| Phase | Epochs | τ (threshold) | Learning Rate | Write Rate Target |
|-------|--------|---------------|---------------|-------------------|
| Warmup | 1–2 | 0.3 | 1e-4 | 40–50% |
| Stabilization | 3–5 | 0.5 | 5e-5 | 20–30% |
| Fine-tuning | 6–10 | 0.7 | 1e-5 | 10–15% |

---

## Data Preparation

The LoCoMo dataset (`locomo10.json`) contains 19 long conversation sessions and 100+ multiple‑choice questions. Run preprocessing:

```angular2html
# Create necessary directories
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "outputs\checkpoints"
New-Item -ItemType Directory -Force -Path "outputs\logs"
New-Item -ItemType Directory -Force -Path "outputs\traces"
New-Item -ItemType Directory -Force -Path "outputs\results"

# Process data
python data/preprocess.py --input data/raw/locomo10.json --output data/processed/

# Verify processed data
python -c "
import json
data = json.load(open('data/processed/train.json', 'r', encoding='utf-8'))
print(f'Examples: {len(data)}')
print(f'Sample: {data[0][\"question\"][:80]}...')
"
```

This creates `train.json`, `val.json`, `test.json` (70/15/15 split). Each example includes:
- `session_texts`: concatenated conversation history
- `question`: multiple-choice question
- `choices`: list of 10 options (A–J)
- `answer_index`: correct answer (0-9)
- `question_type`: single-hop, multi-hop, temporal, entity tracking, adversarial, open_domain

---

## Training

### Quick Setup Test
```angular2html
# Test GPU and model loading
python scripts/test_setup.py
```

### Optimized Training (Recommended - 5x faster)

The optimized training uses LoRA to train only ~30M parameters instead of 3B, reducing epoch time from 1 hour to ~15 minutes:

```angular2html
# Train for 5 epochs (takes ~1-2 hours total)
python src/training/train_3b_optimized_full.py --epochs 5

# Train for 10 epochs (takes ~2-3 hours total)
python src/training/train_3b_optimized_full.py --epochs 10
```

### Resume Training from Checkpoint

```angular2html
# Train first 5 epochs
python src/training/train_3b_resume.py --epochs 5

# Resume to train 5 more (total 10)
python src/training/train_3b_resume.py --resume outputs/checkpoints/epoch_5 --epochs 10
```

### Quick Test (Smoke Test)
```angular2html
# Quick test with 3 sessions
python src/training/train_3b.py --config config/rtx3060_3b.yaml --quick_test --max_sessions 3
```

### Training Features:
- **LoRA fine-tuning**: Only trains 29.9M parameters (vs 3B)
- **Mixed precision**: FP16 for faster training
- **Gradient accumulation**: Simulates larger batches
- **Learning rate scheduling**: Warmup + linear decay
- **Checkpoint saving**: Every epoch with best model tracking

**Monitor GPU usage:**
```angular2html
nvidia-smi -l 2
```

---

## Evaluation

### Test the Trained Model
```angular2html
# Test trained LoRA model on sample questions
python scripts/test_trained_model.py
```

### Comprehensive Evaluation with F1 and BLEU Scores
```angular2html
# Full evaluation on test set (calculates F1, BLEU, Accuracy)
python scripts/final_evaluation.py
```

### Evaluate using the trained LoRA model (default: outputs/best_lora_model)
```angular2html
python scripts/final_evaluation_with_lora.py
```

### Quick Baseline Evaluation (No LLM)
```angular2html
# Random and majority baselines
python scripts/quick_metrics.py
```

### Evaluate with Checkpoint
```angular2html
# Convert checkpoint if needed
python scripts/convert_checkpoint.py

# Run evaluation
python scripts/run_experiment.py --mode eval --checkpoint outputs/checkpoints/best_model_3b.pt
```

---

## Metrics & Results

### Evaluation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Accuracy** | % correct answers | >65% after 5 epochs |
| **F1 Score** | Token-level overlap | >0.60 after 5 epochs |
| **BLEU Score** | N-gram overlap | >0.50 after 5 epochs |
| **Explicit Write Rate** | % steps where u_t ≥ τ | 10–15% |
| **Uncertainty Calibration (ECE)** | Expected Calibration Error | <0.1 |

### Expected Learning Curve (5 Epochs)

| Epoch | Accuracy | F1 Score | BLEU Score | Time |
|-------|----------|----------|------------|------|
| 0 (Baseline) | 25% | 0.00 | 0.00 | - |
| 1 | 35-40% | 0.35 | 0.30 | 15 min |
| 2 | 45-50% | 0.45 | 0.40 | 15 min |
| 3 | 55-60% | 0.55 | 0.48 | 15 min |
| 4 | 60-65% | 0.60 | 0.52 | 15 min |
| 5 | 65-70% | 0.65 | 0.55 | 15 min |

### Results by Question Type (After 5 Epochs)

| Type | Accuracy | F1 Score | Samples |
|------|----------|----------|---------|
| Single Hop | 70-75% | 0.70 | 43 |
| Multi Hop | 60-65% | 0.58 | 42 |
| Temporal | 65-70% | 0.62 | 14 |
| Open Domain | 60-65% | 0.55 | 116 |
| Adversarial | 55-60% | 0.50 | 84 |
| **Overall** | **65-70%** | **0.60** | **299** |

### Sample Output

```
Epoch 5 completed in 14.5 min
  Train Loss: 0.8234
  Val Accuracy: 67.34%
  Per-type Accuracy:
    adversarial: 58.33%
    multi_hop: 61.90%
    open_domain: 62.07%
    single_hop: 74.42%
    temporal_reasoning: 64.29%
  ✓ Saved best model (accuracy: 67.34%)
```

---

## Quick Reference

```angular2html
# Setup test
python scripts/test_setup.py

# Quick test (3 sessions)
python src/training/train_3b.py --config config/rtx3060_3b.yaml --quick_test --max_sessions 3

# Optimized training (5 epochs, ~1-2 hours)
python src/training/train_3b_optimized_full.py --epochs 5

# Optimized training (10 epochs, ~2-3 hours)
python src/training/train_3b_optimized_full.py --epochs 10

# Resume training from checkpoint
python src/training/train_3b_resume.py --resume outputs/checkpoints/epoch_5 --epochs 10

# Test trained model
python scripts/test_trained_model.py

# Full evaluation with F1 and BLEU
python scripts/final_evaluation.py

# Evaluate using the trained LoRA model (default: outputs/best_lora_model)
python scripts/final_evaluation_with_lora.py

# Quick baseline metrics (random + majority)
python scripts/quick_metrics.py

# Convert checkpoint format
python scripts/convert_checkpoint.py

# GPU monitoring
nvidia-smi -l 2

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
```


---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of Memory (OOM)** | Use `batch_size=1`. Reduce `max_sessions` to 3. Close other GPU apps. |
| **Slow training (>1 hour/epoch)** | Use optimized training: `python src/training/train_3b_optimized_full.py --epochs 5` |
| **Low accuracy (<50%)** | Train for more epochs (10). Check data preprocessing. Use LoRA with r=16. |
| **Zero accuracy (0%)** | Model output parsing issue. Use `scripts/final_evaluation.py` which has fixed parsing. |
| **Checkpoint loading error** | Run `python scripts/convert_checkpoint.py` to fix format. |
| **CUDA not available** | Reinstall PyTorch with correct CUDA version. Check `nvidia-smi`. |
| **BitsAndBytes errors** | `pip install bitsandbytes==0.41.3`. Ensure CUDA toolkit matches. |

### GPU Memory Usage

| Configuration | VRAM Used | Time/Epoch |
|---------------|-----------|------------|
| Full model (3B) | 8-10 GB | ~60 min |
| LoRA (optimized) | 6-8 GB | ~15 min |
| LoRA + batch_size=1 | 5-7 GB | ~12 min |

---

## Citation

If you use AdaptiveBelief in your research, please cite:

```bibtex
@misc{adaptivebelief2025,
  author = {AdaptiveBelief Team},
  title = {AdaptiveBelief: Uncertainty-Gated Implicit-Explicit Memory for Long-Horizon AI Agents},
  year = {2025},
  note = {Built with Qwen2.5-3B-Instruct on LoCoMo},
  howpublished = {\url{https://github.com/adaptivebelief/AdaptiveBelief}}
}
```

---

## License

MIT License

Copyright (c) 2025 AdaptiveBelief Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

## Contact & Support

- **Issues**: Please open a GitHub issue
- **Questions**: Email the AdaptiveBelief team
- **Contributions**: Pull requests welcome

---

*All source files are fully implemented and tested on RTX 3060 12GB VRAM. Optimized training reduces epoch time from 1 hour to ~15 minutes using LoRA.*


```angular2html
# Evaluate using the trained LoRA model (default: outputs/best_lora_model)
python scripts/final_evaluation_with_lora.py

# Evaluate with a specific LoRA checkpoint
python scripts/final_evaluation_with_lora.py --lora_path outputs/checkpoints/epoch_5

# Evaluate only 50 samples for quick testing
python scripts/final_evaluation_with_lora.py --max_samples 50
```
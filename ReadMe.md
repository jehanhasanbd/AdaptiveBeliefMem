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
git clone https://github.com/jehanhasanbd/AdaptiveBeliefMem
cd AdaptiveBeliefMem

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
│   │   ├── train_3b_optimized_full.py  # Full optimized training with LoRA
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
│   ├── final_evaluation_with_lora.py        # Comprehensive final evaluation with LoRA attach
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

```angular2html
# Quick test (3 sessions)
python src/training/train_3b.py --config config/rtx3060_3b.yaml --quick_test --max_sessions 3

# Optimized training (5 epochs)
python src/training/train_3b_optimized_full.py --epochs 5
```
---

## Evaluation


### Comprehensive Evaluation with F1 and BLEU Scores
```angular2html
# Full evaluation on test set (calculates F1, BLEU, Accuracy)
python scripts/final_evaluation.py
```

### Evaluate using the trained LoRA model (default: outputs/best_lora_model)
```angular2html
python scripts/final_evaluation_with_lora.py
```



## Quick Reference

```angular2html

# Full evaluation with F1 and BLEU
python scripts/final_evaluation.py

# Evaluate using the trained LoRA model (default: outputs/best_lora_model)
python scripts/final_evaluation_with_lora.py

# Evaluate using the trained LoRA model (default: outputs/best_lora_model)
python scripts/final_evaluation_with_lora.py

# Evaluate with a specific LoRA checkpoint
python scripts/final_evaluation_with_lora.py --lora_path outputs/checkpoints/epoch_5

# Evaluate only 50 samples for quick testing
python scripts/final_evaluation_with_lora.py --max_samples 50

```

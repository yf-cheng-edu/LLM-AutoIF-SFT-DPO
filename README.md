<div align="center">

# AutoIF-LLM

**Optimizing Domain Models via AutoIF Data Synthesis and SFT+DPO Alignment**

*Based on [Self-play with Execution Feedback (ICLR 2025 Spotlight)](https://arxiv.org/abs/2406.13542)*

---

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Student%20Model-Qwen2.5--1.5B-purple)](https://huggingface.co/Qwen)
[![Powered by](https://img.shields.io/badge/Teacher%20Model-DeepSeek%20API-00BFFF)](https://platform.deepseek.com/)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Domain Adaptation](#domain-adaptation)
- [Architecture](#architecture)
- [Training Metrics](#training-metrics)
- [Evaluation Results](#evaluation-results)
- [Pipeline Statistics](#pipeline-statistics)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

**AutoIF-LLM** is a fully automated fine-tuning framework that enhances the instruction-following capabilities of large language models (LLMs) through **execution feedback** and **self-play**. Leveraging a teacher-student architecture and a multi-stage data synthesis pipeline, AutoIF generates high-quality Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) data from a minimal set of seed instructions — **with zero human annotation required**.

The framework uses the **DeepSeek API** as the teacher model and fine-tunes a lightweight student model (**Qwen2.5-1.5B-Instruct**) via **LLaMA-Factory** on a single GPU, eliminating the need for multi-node infrastructure.

AutoIF is designed to be **domain-agnostic**: simply swapping the seed instruction file produces specialized models for over 30 vertical domains, including law, finance, medicine, and education.

> **Design Rationale:** AutoIF converts execution feedback from large models into a scalable alignment signal. Negative samples filtered out during SFT become high-contrast rejected examples for DPO training, maximizing the margin signal required for effective preference learning.

---

## Key Features

- **Domain-Agnostic Pipeline** — Replace a single seed instruction file to target any professional domain. Includes 30+ built-in domain templates out of the box.
- **Fully Automated Workflow** — End-to-end execution from raw seed instructions to a fine-tuned model, with zero human labeling or review.
- **Single-GPU Compatible** — The complete workflow runs on a single NVIDIA A800 (80 GB), significantly lowering the hardware barrier to entry.
- **Execution-Verified Data Quality** — A Python-based automated validator filters training samples via real code execution, achieving a ~71.7% rejection rate to ensure only logically consistent, constraint-satisfying samples are retained.
- **Two-Stage Alignment** — Combines SFT (high-score filtering, threshold > 9) with DPO (chosen/rejected pair construction, threshold > 0.5) for precise constraint-following behavior.
- **Flexible Query Generation** — Supports both ShareGPT-style query augmentation and teacher-model-simulated response generation.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Inference Engine** | vLLM 0.5.5 |
| **Training Framework** | LLaMA-Factory |
| **Teacher Model** | DeepSeek API |
| **Student Model** | Qwen2.5-1.5B-Instruct (3 GB) |
| **NLI Filtering Model** | mDeBERTa-v3-base (2.5 GB) |
| **Fine-tuning Method** | LoRA (SFT + DPO) |
| **Compute Precision** | BF16 |
| **Runtime Environment** | Python 3.10+, CUDA 12.x, PyTorch 2.4.0 |

---

## Installation

### Prerequisites

| Requirement | Specification |
|---|---|
| **GPU** | NVIDIA A800 (80 GB VRAM) or equivalent |
| **Operating System** | Ubuntu 20.04 / 22.04 |
| **Python** | 3.10+ |
| **CUDA** | 12.x |
| **Core Dependencies** | PyTorch 2.4.0, vLLM 0.5.5 (pinned versions) |
| **Disk Space** | ≥ 40 GB available |
| **DeepSeek API Key** | Required for the data synthesis stage (teacher model) |

### Step 1 — Clone and Configure the Environment

```bash
bash scripts/setup.sh

```

This script automatically:

1. Installs PyTorch 2.4.0 with CUDA 12.1 support.
2. Installs vLLM 0.5.5 for inference acceleration.
3. Installs LLaMA-Factory as the backend training framework.

### Step 2 — Download Student and Filtering Models

```bash
bash scripts/download_models.sh

```

Downloads the student model (Qwen2.5-1.5B-Instruct) and the NLI filtering model (mDeBERTa-v3).

### Step 3 — Configure the API Key

Data synthesis relies on the DeepSeek API. Configure your key before running any pipeline:

```bash
export SUPERVISOR_API_KEY="YOUR_DEEPSEEK_API_KEY"

```

---

## Quick Start

### Option A: One-Command Full Pipeline

The `run_all.sh` orchestration script automatically chains data synthesis, SFT, DPO, and vLLM deployment testing. It seamlessly supports domain switching.

```bash
# General domain (default)
bash scripts/run_all.sh

# Domain-specific fine-tuning
bash scripts/run_all.sh --domain 法律
bash scripts/run_all.sh --domain 金融
bash scripts/run_all.sh --domain 医疗

# Run in the background with log monitoring
nohup bash scripts/run_all.sh --domain 法律 > run.log 2>&1 &
tail -f run.log

```

> **Note:** This script automatically handles LLaMA-Factory dataset registration and applies the Qwen `rope_scaling` compatibility patch before deployment.

#### Pipeline Stages

| Stage | Description |
| --- | --- |
| **Stage 1** | AutoIF 9-step SFT data synthesis (via DeepSeek API) |
| **Stage 2** | DPO preference pair construction |
| **Stage 3** | SFT training with LoRA |
| **Stage 4** | LoRA weight merging (SFT) |
| **Stage 5** | DPO training with LoRA |
| **Stage 6** | LoRA weight merging (DPO) |
| **Stage 7** | Environment compatibility patches |
| **Stage 8** | Offline inference validation and testing with vLLM |

---

### Option B: Step-by-Step Manual Execution

For users who wish to inspect pipeline internals or perform targeted debugging, each stage can be executed independently.

#### Stage 1 — Data Synthesis (AutoIF)

The synthesis stage calls the DeepSeek API and constructs SFT data across 9 steps (RFT augmentation, validator function generation, back-translation filtering, etc.) and DPO data across 3 steps.

```bash
# SFT data construction (Steps 1–9)
python code_sft/1_RFT.py
# ... execute steps 2 through 8 sequentially ...
python code_sft/9_sft_data_construction.py

# DPO data construction
python code_dpo/1_dpo_rft_wash.py
python code_dpo/2_dpo_data_query_construct.py

```

#### Stage 2 — SFT Fine-Tuning and Weight Merging

Fine-tune Qwen2.5-1.5B using LoRA via LLaMA-Factory.

```bash
cd LlamaFactory

# Fine-tune with LoRA
llamafactory-cli train ../configs/llamafactory_sft_lora.yaml

# Merge LoRA weights into the base model
llamafactory-cli export \
  --model_name_or_path ../models/student/Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ../models/model_d_sft \
  --export_dir ../models/model_d_sft_merged \
  --finetuning_type lora \
  --template qwen

cd ..

```

#### Stage 3 — DPO Alignment and Weight Merging

Perform preference reinforcement learning on top of the merged SFT model.

```bash
cd LlamaFactory

# DPO training on top of the merged SFT model
llamafactory-cli train ../configs/llamafactory_dpo_lora.yaml

# Merge DPO weights (using best-converging checkpoint-175)
llamafactory-cli export \
  --model_name_or_path ../models/model_d_sft_merged \
  --adapter_name_or_path ../models/model_d_dpo/checkpoint-175 \
  --export_dir ../models/model_d_dpo_merged \
  --finetuning_type lora \
  --template qwen

cd ..

```

#### Stage 4 — Compatibility Patches and Evaluation

Apply patches to resolve vLLM compatibility issues that may arise after Qwen model merging.

```bash
python patches/fix_config.py
python patches/fix_qwen.py models/model_d_dpo_merged
python tests/models_to_test.py

```

#### Stage 5 — vLLM Deployment and API Testing

Serve the final fine-tuned model using the high-throughput inference engine.

```bash
# Launch the vLLM inference server
vllm serve models/model_d_dpo_merged \
  --dtype bfloat16 \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name qwen \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7

# In a separate terminal, run the API test
python tests/test_vllm.py

```

> **Screenshot placeholder** — `images/test_vllm_result.png`

---

## Domain Adaptation

AutoIF's core innovation lies in its ability to **automatically generate high-quality, domain-specific training data** simply by swapping the seed instruction file.

### Built-in Domains (30+)

| Category | Domains |
| --- | --- |
| **Basic Sciences** | Mathematics, Physics, Chemistry, Biology, Astronomy, Geography |
| **Engineering** | Civil Engineering, Mechanical Engineering, Electrical Engineering, Chemical Engineering, Materials Science, Energy Engineering |
| **Humanities & Social Sciences** | Literature, History, Philosophy, Journalism, Sociology, Psychology |
| **Business & Management** | Business Administration, Accounting, Public Administration, E-Commerce, Finance |
| **Applied Domains** | Law, Medicine, Education, Software Development |
| **Arts & Sports** | Fine Arts, Music, Physical Education |

### Custom Domain Configuration

```bash
# List all available built-in domain templates
python scripts/generate_seed_instructions.py --list

# Automatically generate seed instructions for a new custom domain using an LLM
python scripts/generate_seed_instructions.py --domain 建筑设计 --use-llm

# Run the full automated fine-tuning pipeline for the custom domain
bash scripts/run_all.sh --domain 建筑设计

```

> **Requirement:** For custom domain contributions, add the new domain entry to `scripts/extended_domains.py` and provide a minimum of **10 representative seed instructions** for that domain.

---

## Architecture

### Data Synthesis Pipeline (9 Steps)

| Step | Operation | Input → Output |
| --- | --- | --- |
| **Step 1** | RFT Instruction Augmentation | 36 seeds → 830 new instructions |
| **Step 2** | Validator Function Generation | 830 + 36 = 866 instructions |
| **Step 3** | Cross-Validation Filtering | 866 → 491 (pass rate: 56.7%) |
| **Step 4** | Back-Translation | 491 instructions |
| **Step 5** | NLI Consistency Filtering | 491 → 426 (retention rate: 86.76%) |
| **Step 6** | Query Augmentation + Response Generation | 426 × ~10 queries × 5 responses = 21,300 candidates |
| **Step 7** | Execution-Based Quality Scoring | 21,300 → 6,031 (passing Python execution) |
| **Step 8** | High-Quality SFT Sample Selection (Score > 9) | 6,031 → 2,239 SFT samples |
| **Step 9** | SFT Dataset Construction | Output: `IF_sft_data.json` |

> **Screenshot placeholder** — Score distribution visualization: `images/score_distribution.png`

### DPO Preference Pair Construction

The DPO pipeline **intentionally bypasses** the Step 8 high-score filter and traces back to the full response pool from Step 6. Samples filtered out during SFT become high-contrast negative examples that maximize the margin differential required for effective DPO preference learning.

**DPO Step 1 (`1_dpo_rft_wash.py`):**

* Re-processes the 4,260 response candidates from Step 6.
* Computes pass rate for each response using the validator function.
* Output format: `[response_text, accuracy_score]`.

**DPO Step 2 (`2_dpo_data_query_construct.py`):**

1. **Separate positive/negative samples:** Responses with pass rate ≥ 0.5 → `chosen`; pass rate = 0 → `rejected`.
2. **Pairing condition:** A valid pair requires at least one `chosen` and one `rejected` response under the same prompt.
3. **Combination sampling:** Sample up to 2 `chosen` and up to 2 `rejected` responses per prompt, generating all valid positive-negative pairings.

### Training Hyperparameters

| Hyperparameter | SFT Stage | DPO Stage |
| --- | --- | --- |
| **Fine-tuning Method** | LoRA (rank=16, α=32) | LoRA (rank=16, α=32) |
| **Learning Rate** | 5e-5 | 5e-6 |
| **Epochs** | 3.0 | 2.0 |
| **Max Sequence Length** | 2048 | 2048 |
| **Compute Precision** | BF16 | BF16 |
| **Batch Size (per device / grad acc)** | 4 / 4 | 2 / 8 |
| **Eval & Save Frequency** | Every 150 steps | Every 25 steps |
| **LR Scheduler** | Cosine (warmup_ratio=0.05) | Cosine (warmup_ratio=0.1) |
| **LoRA Target Modules** | q, k, v, o, gate, up, down | q, k, v, o, gate, up, down |
| **Beta (DPO)** | — | 0.3 |

---

## Training Metrics

### SFT Convergence

SFT training proceeds stably without overfitting. Validation loss decreases smoothly from 1.41 to 1.20, stabilizing at approximately step 350.

> **Screenshot placeholder** — SFT Training Loss: `images/SFT/training_loss.png`
> **Screenshot placeholder** — SFT Eval Loss: `images/SFT/training_eval_loss.png`

### DPO Preference Alignment

Over 2 epochs of DPO training, reward accuracy (`Rewards/Accuracies`) climbs steadily and stabilizes at approximately **83%**. The evaluation loss reaches its minimum at step 175 without rebounding; accordingly, **`checkpoint-175`** is selected as the final production checkpoint.

> **Screenshot placeholder** — DPO Rewards/Accuracies: `images/DPO/dpo_training_rewards_accuracies.png`
> **Screenshot placeholder** — DPO Eval Loss: `images/DPO/dpo_training_eval_loss.png`
> **Screenshot placeholder** — DPO Training Loss: `images/DPO/dpo_training_loss.png`

---

## Evaluation Results

The following comparisons demonstrate constraint-following behavior across the base model, SFT-aligned model, and DPO-aligned model on a high-difficulty instruction-following benchmark.

### 1. Baseline: Complete Constraint Failure

Prior to alignment training, the unmodified base model fails multiple simultaneous constraints — including format, character set, and lexical boundary constraints — within a single inference run.

> **Screenshot placeholder** — `images/base/base_response.png`

### 2. Progressive Alignment Success

**SFT Stage (Initial Compliance):** The model successfully captures target format constraints (e.g., all-uppercase output with STOP markers).

> **Screenshot placeholder** — `images/SFT/SFT_response.png`

**DPO Stage — Checkpoint 175 (Full Constraint Adherence):** Through probabilistic alignment via preference pairs, the model fully internalizes all constraints (telegram format, exact three-sentence limit, all sentences beginning with 'B', all sentences beginning with 'T').

> **Screenshot placeholder** — `images/DPO/DPO_response.png`

---

## Pipeline Statistics

The following figures represent real production data from a benchmark general-domain fine-tuning run on AutoDL (NVIDIA A800, 80 GB).

> **Reproducibility Note:** Due to stochastic generation in LLMs and random sampling during DPO pair construction, absolute counts may vary slightly across independent runs.

### End-to-End Data Flow

| Stage | Metric | Count | Notes |
| --- | --- | --- | --- |
| **Step 0** | Raw seed instructions | 36 | Initial seed set |
| **Step 1** | RFT-augmented instructions | 830 | Expanded instruction pool |
| **Step 2** | Instructions entering validator construction | 866 | 36 original + 830 augmented |
| **Step 3** | Cross-validation survivors | 491 | Pass rate: 56.7% |
| **Step 5** | NLI consistency filter survivors | 426 | Retention rate: 86.76% |
| **Step 6** | Total query/response candidates | 21,300 | 426 × ~10 queries × 5 responses |
| **Step 7** | Execution-validated samples | 6,031 | Python execution filter (Accuracy > 0) |
| **Step 8** | High-quality SFT samples | 2,239 | Composite score > 9 / 10 |
| **DPO** | Final DPO preference pairs | 2,159 | Meeting chosen/rejected threshold (≥ 0.5) |

### Execution Filter Rejection Rate (Step 7)

Of the 21,300 candidate responses generated in Step 6:

* **Passed Python execution validation:** 6,031 samples
* **Rejected by Python validator:** 15,269 samples
* **Execution filter rejection rate: 71.7%**

---

## Project Structure

```text
AutoIF-LLM/
├── .gitignore
├── README.md
├── requirements.txt               # Pinned Python dependencies
├── code_dpo/                      # DPO preference pair pipeline
│   ├── 1_dpo_rft_wash.py
│   └── 2_dpo_data_query_construct.py
├── code_sft/                      # AutoIF data synthesis pipeline
│   ├── 1_RFT.py
│   ├── ...
│   ├── 9_sft_data_construction.py
│   └── utils.py                   # Core environment variables and configuration constants
├── configs/                       # LLaMA-Factory training configurations
│   ├── llamafactory_sft_lora.yaml
│   ├── llamafactory_dpo_lora.yaml
│   ├── llama_factory_dataset_info.json
│   └── pipeline_config.yaml
├── images/                        # Documentation and evaluation result assets
│   ├── base/
│   ├── DPO/
│   └── SFT/
├── output/                        # Runtime data output directory
│   ├── dpo_pairs_flat.jsonl
│   └── IF_sft_data.json
├── patches/                       # Environment compatibility patches
│   ├── dpo2_patches.py
│   ├── fix_config.py
│   └── fix_qwen.py
├── sample_data/
│   └── seed_instruction.txt       # Default seed instruction file
├── scripts/                       # Automation and environment initialization scripts
│   ├── download_models.sh
│   ├── extended_domains.py
│   ├── generate_seed_instructions.py
│   ├── run_all.sh
│   └── setup.sh
├── tests/                         # Test and deployment validation scripts
│   ├── models_to_test.py
│   └── test_vllm.py
└── tools/
    └── view_scores.py             # Data quality score visualization utility

```

---

## Troubleshooting

Due to rapid updates in upstream dependencies, version conflicts may occasionally arise. AutoIF includes three built-in compatibility patches that `run_all.sh` applies automatically during full pipeline execution.

| Script | Trigger Condition | Resolution | Manual Execution |
| --- | --- | --- | --- |
| **`fix_qwen.py`** | vLLM fails to start with `rope_scaling` validation error | Injects `{"factor": 1.0, "type": "default"}` into the model's `config.json` | `python patches/fix_qwen.py ./models/model_d_dpo_merged` |
| **`fix_config.py`** | Training framework cannot parse non-standard positional encoding fields | Traverses all model configs under `models/` and removes incompatible parameters | `python patches/fix_config.py` |
| **`dpo2_patches.py`** | Nested ShareGPT conversation arrays cause parsing errors during training | Flattens DPO data into the more stable Alpaca format and re-registers the dataset | `python patches/dpo2_patches.py` |

---

## Contributing

Contributions are welcome. Please follow the standard GitHub workflow:

1. **Fork** this repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with clear, descriptive commit messages.
4. Open a **Pull Request** targeting the `main` branch.

For domain template contributions, add your new entry to `scripts/extended_domains.py` and include a minimum of **10 representative seed instructions** for the target domain.

---

## Citation

If you use AutoIF in academic research or build upon this project, please cite the original paper:

```bibtex
@article{dong2024self,
  title={Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models},
  author={Dong, Guanting and Lu, Keming and Li, Chengpeng and Xia, Tingyu and Yu, Bowen and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2406.13542},
  year={2024}
}

```


## License

This project is released under the **Apache License 2.0**.

Downstream foundation models used by this project (e.g., Qwen2.5 series, mDeBERTa-v3) are subject to their respective original licenses. Please review those licenses carefully before any commercial use.

```

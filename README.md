<div align="center">

# AutoIF-LLM

**Domain-Specific Model Optimization via AutoIF Data Synthesis and SFT+DPO Alignment**

*Based on: [Self-play with Execution Feedback (ICLR 2025 Spotlight)](https://arxiv.org/abs/2406.13542)*

---

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Student%20Model-Qwen2.5--1.5B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Powered by](https://img.shields.io/badge/Teacher%20Model-DeepSeek--V4--Flash-00BFFF)](https://platform.deepseek.com/)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack & Requirements](#tech-stack--requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Domain Adaptation](#domain-adaptation)
- [System Architecture](#system-architecture)
- [Training Metrics & Evaluation](#training-metrics--evaluation)
- [Pipeline Statistics](#pipeline-statistics)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

**AutoIF-LLM** is a fully automated fine-tuning framework designed to significantly improve the instruction-following capabilities of large language models (LLMs) through **Execution Feedback** and **Self-Play**. The framework adopts a **Teacher-Student** architecture combined with a multi-stage data synthesis pipeline, capable of automatically generating high-quality Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) data from a minimal set of seed instructions — **entirely without human annotation**.

The framework offers strong **cross-domain generalizability**: simply swap the seed instruction file to automatically generate domain-specific fine-tuning data for any of 30+ vertical industries (e.g., legal, finance, healthcare).

---

## Key Features

- **Cross-Domain Adaptability** — Out-of-the-box support for 30+ built-in vertical domains; swap the seed file to auto-generate domain fine-tuning data.
- **Fully Automated Workflow** — End-to-end automation from raw seed instructions to final INT4 quantization and deployment.
- **Single-GPU Lightweight** — The complete workflow runs smoothly on a single NVIDIA A800 (80 GB).
- **Execution-Verified Quality** — A built-in automated Python code execution validator filters out over 70% of logically inconsistent samples.
- **Two-Stage Precision Alignment** — Combines SFT (high-scoring samples with composite score > 9) and DPO (high-contrast failure samples as rejected pairs) for precise instruction constraint following.

---

## Tech Stack & Requirements

| Category | Details |
| --- | --- |
| **Core Models** | **Teacher:** DeepSeek-V4-Flash <br> **Student:** Qwen2.5-1.5B-Instruct <br> **Auxiliary:** mDeBERTa-v3-base |
| **Fine-tuning Framework** | LLaMA-Factory (LoRA SFT + DPO) |
| **Quantization & Deployment** | Auto-GPTQ (INT4), vLLM 0.5.5 |
| **Hardware Requirements** | NVIDIA A800 (80 GB VRAM) or equivalent; ≥ 40 GB free disk space |
| **System & Environment** | Ubuntu 20.04/22.04, Python 3.10+, CUDA 12.x |

> **Note:** To avoid dependency conflicts, this project uses a **dual-environment architecture**. Base training runs in the `base` environment, while INT4 quantization and vLLM deployment are handled in a separate `gptq_env` virtual environment.

---

## Installation

### Step 1 — Initialize the Training Environment

Run the one-click setup script to automatically install base training dependencies, the LLaMA-Factory framework, and register dataset configurations:

```bash
bash scripts/setup.sh
```

### Step 2 — Download Models Locally

Run the model download script to fetch the student model (Qwen2.5-1.5B-Instruct) and the NLI filtering model (mDeBERTa-v3):

```bash
bash scripts/download_models.sh
```

### Step 3 — Configure the Teacher API Key

Data synthesis relies heavily on the DeepSeek API. Set your API key in the terminal:

```bash
export SUPERVISOR_API_KEY="YOUR_DEEPSEEK_API_KEY"
```

---

## Quick Start

### Option A: One-Click Full Pipeline

Use the orchestration script `run_all.sh` to automatically chain data synthesis, SFT, DPO, INT4 quantization, and vLLM deployment testing in one command, with native support for vertical domain switching.

```bash
# General-domain training
bash scripts/run_all.sh

# Domain-specific fine-tuning
bash scripts/run_all.sh --domain legal
bash scripts/run_all.sh --domain finance

# Run in background with live log monitoring
nohup bash scripts/run_all.sh --domain legal > run.log 2>&1 &
tail -f run.log
```

> **Note:** This script includes an automatic environment-switching mechanism. During Stage 7 (quantization), it will automatically activate the `gptq_env` virtual environment. Make sure you have created this environment in advance (see Option B — Stage 5).

#### Pipeline: 10 Automated Stages at a Glance

- **Stage 1:** AutoIF data synthesis (9-step SFT construction + 3-step DPO construction and flattening).
- **Stage 2 & 3:** SFT supervised fine-tuning and LoRA weight merging.
- **Stage 4 & 5:** DPO preference alignment training and LoRA weight merging.
- **Stage 6:** Offline comparison of Base / SFT / DPO model outputs.
- **Stage 7:** Switch to `gptq_env` and perform GPTQ INT4 model quantization.
- **Stage 8:** Launch vLLM INT4 local service and run automated API tests.
- **Stage 9:** Extract 200 high-quality test samples from the synthesized dataset and run quantitative evaluation across Base / SFT / DPO / GPTQ four models using Transformers batch inference and vLLM.
- **Stage 10:** Invoke DeepSeek LLM-as-a-Judge to conduct pairwise head-to-head comparisons across all four models and output a comprehensive win-rate report.

---

### Option B: Step-by-Step Manual Execution

If you want to inspect intermediate artifacts or perform targeted debugging, you can run each stage independently.

#### Stage 1 — Data Synthesis (AutoIF)

This stage calls the DeepSeek API to build SFT data across 9 steps and DPO data across 3 steps.

```bash
# SFT data construction (Steps 1–9)
python code_sft/1_RFT.py
# ... run steps 2 through 8 sequentially ...
python code_sft/6_concat_sharegpt_query.py
# Extract 200 high-quality samples (with validator functions and at least one perfect-score response)
# as a standardized test set for quantitative evaluation:
python tools/extract_test_set.py
# ...
python code_sft/9_sft_data_construction.py

# DPO data construction
python code_dpo/1_dpo_rft_wash.py
python code_dpo/2_dpo_data_query_construct.py
```

#### Stage 2 — SFT Fine-Tuning & Weight Merging

LoRA fine-tuning via LLaMA-Factory.

```bash
cd LlamaFactory

# Launch LoRA fine-tuning
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

#### Stage 3 — DPO Alignment & Weight Merging

Preference reinforcement learning alignment on top of the merged SFT model.

```bash
cd LlamaFactory

# DPO training based on the merged SFT model
llamafactory-cli train ../configs/llamafactory_dpo_lora.yaml

# Merge DPO weights (select the best-converged checkpoint-175)
llamafactory-cli export \
  --model_name_or_path ../models/model_d_sft_merged \
  --adapter_name_or_path ../models/model_d_dpo_2/checkpoint-175 \
  --export_dir ../models/model_d_dpo_merged \
  --finetuning_type lora \
  --template qwen

cd ..
```

#### Stage 4 — Compatibility Patches & Evaluation

Apply Qwen-related compatibility patches and test text generation:

```bash
python patches/fix_config.py
python patches/fix_qwen.py models/model_d_dpo_merged
python tests/models_to_test.py
```

#### Stage 5 — Virtual Environment Setup & GPTQ INT4 Quantization

Because quantization and deployment tools (such as `vllm` and `auto-gptq`) have strict dependency requirements, **you must use a separate Conda virtual environment and install dependencies via the provided requirements file**.

```bash
# 1. Create and activate the virtual environment
conda create -n gptq_env python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate gptq_env

# 2. Install the prebuilt inference environment
pip install -r requirements_gptq_vllm.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. Force source compilation of auto-gptq for A800 architecture (resolves kernel incompatibility)
pip uninstall auto-gptq -y
BUILD_CUDA_EXT=1 pip install auto-gptq -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. Run INT4 quantization
python tests/GPTQ.py
```

#### Stage 6 — vLLM Deployment

Generate the chat template and start the vLLM backend under `gptq_env`.

```bash
# Generate the Qwen ChatML conversation template
cat << 'EOF' > configs/chatml.jinja
{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}
EOF

# Start the vLLM backend (ensure you are in the gptq_env environment)
vllm serve models/model_d_dpo_merged_gptq_int4 \
    --quantization gptq \
    --dtype float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name qwen \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --chat-template configs/chatml.jinja
```

In a new terminal, activate the environment and run the API test client:

```bash
eval "$(conda shell.bash hook)"
conda activate gptq_env
python tests/test_vllm.py
```

<div align="center">
  <img src="images/test_vllm_result.png" width="800" alt="vLLM Inference Test Results">
</div>

#### Stage 7 — Multi-Model Quantitative Evaluation

**Option A: Transformers Batch Inference (base environment, evaluating Base / SFT / DPO)**

```bash
python tests/evaluate_hf_batched.py
```

**Option B (Optional): Transformers Batch Inference for GPTQ Model (requires a separate `hf_eval` environment)**

Due to dependency conflicts between the GPTQ quantization library and the base environment, a separate environment is needed:

```bash
conda create -n hf_eval python=3.10 -y
conda activate hf_eval
pip install -r requirments_GPTQ_model_hf_eval.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# In tests/evaluate_hf_batched.py, update models_to_test to keep only GPTQ-Model, then run:
python tests/evaluate_hf_batched.py
```

> ⚠️ **Note:** GPTQ under native Transformers runs at approximately 45 tokens/s, significantly slower than vLLM (1482 tokens/s). This is caused by extra computational graph overhead and memory access bottlenecks — this is expected behavior. **Option C is recommended.**

**Option C: vLLM High-Concurrency Batch Evaluation (gptq_env, evaluating all four models in one run)**

```bash
conda activate gptq_env
python tests/evaluate_vllm.py
```

> vLLM delivers approximately **33x** faster inference on GPTQ models compared to native Transformers, and all four models can be evaluated in sequence within the same environment.

#### Stage 8 — LLM-as-a-Judge Comprehensive Comparison

After evaluation, use DeepSeek as the judge model to conduct pairwise head-to-head comparisons across all four models, comprehensively assessing instruction-following strictness and response quality:

```bash
# Ensure you have replaced YOUR_API_KEY with a real DeepSeek Key in llm_judge_all.py
python tools/llm_judge_all.py
# Output: output/all_models_judge_results.json
```

---

## Domain Adaptation

Simply swap the seed instruction file to automatically generate training data for any specific vertical domain.

### Custom Domain Configuration

```bash
# List all available built-in domain templates (30+)
python scripts/generate_seed_instructions.py --list

# Automatically generate seed instructions for a new domain using LLM
python scripts/generate_seed_instructions.py --domain architecture-design --use-llm

# Launch the fully automated fine-tuning pipeline for this domain
bash scripts/run_all.sh --domain architecture-design
```

> **Contribution Requirement:** To contribute a new industry domain template, add it to `scripts/extended_domains.py` with at least 10 representative seed instructions.

---

## System Architecture

### Data Synthesis Pipeline (9-Step Method)

| Step | Description | Sample Funnel |
| --- | --- | --- |
| **Steps 1–2** | RFT instruction expansion and validator function generation | 36 seed instructions → 866 total instructions |
| **Steps 3–5** | Cross-validation, back-translation, and NLI consistency filtering | 866 → 426 instructions retained |
| **Steps 6–7** | Batch response generation and real code execution scoring | 21,300 candidates → 6,031 validated |
| **Steps 8–9** | High-quality selection (composite score > 9) and dataset construction | 6,031 → 2,239 final SFT samples |

<div align="center">
  <img src="images/评分段数据分布统计.png" width="700" alt="AutoIF Data Synthesis Score Distribution">
</div>

### DPO Preference Pair Construction

To maximize the contrast of negative feedback signals, the DPO pipeline traces back to the Step 6 candidate pool: samples that were discarded during SFT for extremely low scores (= 0) are directly repurposed as high-contrast `rejected` negatives. Through rigorous pairing, 2,159 high-quality preference training pairs are synthesized.

### Core Training Hyperparameters

- **Fine-tuning Method:** LoRA (rank=16, α=32, target=q/k/v/o/gate/up/down)
- **SFT Phase:** Learning rate 5e-5, Epochs 3.0, batch size 4 (gradient accumulation 4), cosine schedule.
- **DPO Phase:** Learning rate 5e-6, Epochs 2.0, batch size 2 (gradient accumulation 8), Beta 0.3.

---

## Training Metrics & Evaluation

The following data and figures are from the project's actual training logs, showing metric progression and capability improvements across the SFT and DPO phases.

### 1. Training Curve Convergence

**SFT Phase:** Training loss decreased from 1.65 to 0.94; validation loss smoothly declined from 1.41 to approximately 1.20.

<table align="center">
  <tr>
    <td align="center">
      <img src="images/SFT/training_loss.png" width="380" alt="SFT Training Loss"><br>
      <b>SFT Training Loss</b>
    </td>
    <td align="center">
      <img src="images/SFT/training_eval_loss.png" width="380" alt="SFT Eval Loss"><br>
      <b>SFT Eval Loss</b>
    </td>
  </tr>
</table>

**DPO Phase:** Over 2 epochs of DPO preference training, reward accuracy steadily climbed and stabilized at approximately **83%**. Evaluation loss reached its minimum at step 175 without rebounding; therefore, **`checkpoint-175`** was selected as the final production checkpoint.

<table align="center">
  <tr>
    <td align="center">
      <img src="images/DPO/dpo_training_rewards_accuracies.png" width="250" alt="DPO Rewards Accuracies"><br>
      <b>DPO Reward Accuracy</b>
    </td>
    <td align="center">
      <img src="images/DPO/dpo_training_eval_loss.png" width="250" alt="DPO Eval Loss"><br>
      <b>DPO Eval Loss</b>
    </td>
    <td align="center">
      <img src="images/DPO/dpo_training_loss.png" width="250" alt="DPO Training Loss"><br>
      <b>DPO Training Loss</b>
    </td>
  </tr>
</table>

### 2. Constraint Alignment Progression

We tested three model stages under challenging instructions with concurrent format, character set, and lexical boundary constraints:

**Baseline: All constraints fail**

<div align="center">
  <img src="images/base/base_response.png" width="750" alt="Baseline model output with complete constraint failure">
  <p><i>The base model's response completely fails to follow multiple concurrent constraints</i></p>
</div>

**SFT Stage: Basic constraints captured** (successfully outputs all-uppercase text and appends stop tokens)

<div align="center">
  <img src="images/SFT/SFT_response.png" width="750" alt="SFT model output with initial constraint compliance">
  <p><i>After SFT fine-tuning, the model begins to capture target format constraints</i></p>
</div>

**DPO Stage: Full alignment achieved** (perfectly internalizes all compound constraints, including telegraphic style and initial-letter restrictions)

<div align="center">
  <img src="images/DPO/DPO_response.png" width="750" alt="DPO model output with full constraint compliance">
  <p><i>After DPO preference alignment, Checkpoint 175 fully internalizes all compound constraints</i></p>
</div>

---

### 3. Quantitative Evaluation & LLM Judge Summary

The following data is based on a 200-sample standardized test set extracted from the synthesized dataset. It provides a comprehensive comparison of instruction-following capability progression across training stages and performance benchmarks across different inference backends.

#### 3.1 Instruction-Following Accuracy & Inference Throughput (vLLM vs. Transformers)

The table below summarizes the final results of each model under the vLLM inference framework (see raw benchmark screenshots below):

| Model | Accuracy | Total Time | Batch Throughput (vLLM) |
| --- | --- | --- | --- |
| Base-Model (pre-training baseline) | 19.50% | 7.44 s | 1084.19 tokens/s |
| SFT-Model (after supervised fine-tuning) | 28.50% | 7.05 s | 831.59 tokens/s |
| DPO-Model (after preference alignment) | 37.00% | 7.02 s | 773.42 tokens/s |
| GPTQ-Model (after INT4 quantization) | 36.00% | 8.70 s | **1467.55 tokens/s** |

> 💡 **Key Findings:**
> 1. **Capability gains:** Instruction-following accuracy rises steadily from Base → SFT → DPO. DPO preference alignment achieves a significant improvement of **+17.5 percentage points** over the base model (19.5% → 37%), demonstrating the effectiveness of human preference alignment training.
> 2. **Quantization speedup:** GPTQ INT4 quantization delivers nearly lossless accuracy (36.00%, only 1% drop) while surging inference throughput to **nearly 1500 tokens/s** — approximately **2x** the speed of the aligned model.

**📊 Detailed Performance Reports Comparison:**
*(Note: The comparison reveals that vLLM exhibits a decisive advantage in concurrent inference. Especially for the GPTQ quantized model, Transformers native batch processing speed is only 45.38 tokens/s, while vLLM soars to 1467.55 tokens/s.)*

**1. Base-Model**
<div align="center">
  <img src="./images/base/vll_base.png" width="48%" title="Base vLLM">
  <img src="./images/base/transformer_base.png" width="48%" title="Base Transformers">
</div>

**2. SFT-Model**
<div align="center">
  <img src="./images/SFT/vllm_SFT.png" width="48%" title="SFT vLLM">
  <img src="./images/SFT/transformer_sft.png" width="48%" title="SFT Transformers">
</div>

**3. DPO-Model**
<div align="center">
  <img src="./images/DPO/vllm_DPO.png" width="48%" title="DPO vLLM">
  <img src="./images/DPO/transformer_DPO.png" width="48%" title="DPO Transformers">
</div>

**4. GPTQ-Model (INT4 Quantization)**
<div align="center">
  <img src="./images/GPTQ_model/vllm_GPTQ_model.png" width="48%" title="GPTQ vLLM">
  <img src="./images/GPTQ_model/transformer_GPTQ_model.png" width="48%" title="GPTQ Transformers">
</div>

---

#### 3.2 LLM-as-a-Judge Pairwise Win Rates

In addition to objective instruction-following tests, we conducted blind pairwise comparisons via LLM Judge to further assess overall response quality (tone, coherence, informativeness):

| Matchup | Left Model Win Rate | Right Model Win Rate | Tie Rate |
| --- | --- | --- | --- |
| 【Base】 vs 【SFT】 | 33.50% | **46.50%** | 20.00% |
| 【Base】 vs 【DPO】 | 38.50% | **43.50%** | 18.00% |
| 【Base】 vs 【GPTQ】 | 41.00% | **47.50%** | 11.50% |

**📊 Win-Rate Report Screenshots:**
<div align="center">
  <img src="./images/base_vs_SFT&DPO.png" width="60%" title="Base vs SFT & DPO">
  <br><br> <img src="./images/base_vs_GPTQ_model.png" width="60%" title="Base vs GPTQ">
</div>

> 💡 **Key Findings:**
> - Compared to the baseline, fine-tuned and aligned models (SFT & DPO) hold a significant advantage in direct comparisons.
> - Notably, **the GPTQ quantized model not only avoids a sharp drop in response quality, but achieves a win rate of 47.50% in head-to-head matchups against Base**. This demonstrates that the current INT4 quantization approach excellently preserves the model's generalization capability and semantic coherence — truly delivering both speed and quality.

---

## Pipeline Statistics

*(Based on the general-domain benchmark, running on a single NVIDIA A800 80GB)*

| Pipeline Node | Sample Count | Notes |
| --- | --- | --- |
| Initial seed pool | 36 | Raw data from `seed_instruction.txt` |
| Cross-filtering pass rate | 56.7% | Filters logically conflicting augmented instructions |
| **Executor rejection rate** | **71.7%** | Python execution intercepts non-compliant responses |
| Final SFT samples | 2,239 | High-scoring curated instruction set |
| Final DPO preference pairs | 2,159 | Pairs meeting the chosen/rejected score gap (≥ 0.5) |

---

## Troubleshooting

Due to frequent updates in open-source dependencies, this project includes automated compatibility patches to resolve common crash scenarios. These patches are applied automatically within `run_all.sh`.

| Patch Script | Trigger Condition | Resolution |
| --- | --- | --- |
| **`fix_qwen.py`** | `rope_scaling` parsing error when vLLM starts | Automatically injects safe scaling compatibility fields into the merged model's `config.json`. |
| **`fix_config.py`** | LLaMA-Factory fails to parse positional encoding parameters | Automatically removes non-standard config attributes that block LoRA fine-tuning parsing. |
| **`dpo2_patches.py`** | DPO data source causes array index out-of-bounds in the fine-tuning framework | Flattens nested ShareGPT conversation trees into flat Alpaca dictionary structures. |

---

## Project Structure

```text
AutoIF-LLM/
├── code_dpo/                 # DPO preference pair construction pipeline
├── code_sft/                 # AutoIF 9-step data synthesis pipeline
├── configs/                  # Fine-tuning, quantization, and dataset config files
├── images/                   # Evaluation and data distribution visualizations
├── patches/                  # Environment compatibility patch scripts
├── sample_data/              # Domain-specific seed instructions
├── scripts/                  # Environment setup and full automation workflow scripts
├── tests/                    # Offline validation and vLLM API test scripts
└── tools/                    # Data quality visualization tools
```

---

## Contributing

1. **Fork** this repository and create a feature branch: `git checkout -b feature/your-feature`.
2. Submit descriptive commits and open a **Pull Request** targeting the `main` branch.
3. To contribute an industry domain template, add it to `scripts/extended_domains.py` with ≥ 10 representative seed instructions.

---

## Citation

If you use the AutoIF framework in academic research or if this project inspires your work, please cite the original paper:

```bibtex
@inproceedings{dong2025self,
  title={Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models},
  author={Dong, Guanting and Lu, Keming and Li, Chengpeng and Xia, Tingyu and Yu, Bowen and Zhou, Chang and Zhou, Jingren},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2406.13542}
}
```

---

## License

This project is open-sourced under the **Apache License 2.0**.

Downstream foundation models used in this project (such as the Qwen2.5 series and mDeBERTa-v3) are subject to their respective original open-source licenses. Please carefully review and comply with the relevant model licenses before any commercial use.

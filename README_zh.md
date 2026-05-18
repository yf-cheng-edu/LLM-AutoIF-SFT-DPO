<div align="center">

# AutoIF-LLM

**基于AutoIF数据合成与SFT+DPO对齐的领域模型优化**

*基于论文：[Self-play with Execution Feedback (ICLR 2025 Spotlight)](https://arxiv.org/abs/2406.13542)*

---

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Student%20Model-Qwen2.5--1.5B-purple)](https://huggingface.co/Qwen)
[![Powered by](https://img.shields.io/badge/Teacher%20Model-DeepSeek%20API-00BFFF)](https://platform.deepseek.com/)

</div>

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [领域适配](#领域适配)
- [系统架构](#系统架构)
- [训练指标](#训练指标)
- [评估结果](#评估结果)
- [流水线统计数据](#流水线统计数据)
- [项目结构](#项目结构)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [论文引用](#论文引用)
- [开源许可证](#开源许可证)

---

## 项目概述

**AutoIF-LLM** 是一个全自动化的微调框架，旨在通过**执行反馈（Execution Feedback）**和**自我博弈（Self-Play）**来显著提升大语言模型（LLM）的指令遵循能力。该框架采用“导师-学生（Teacher-Student）”架构以及多阶段数据合成流水线，仅需极少量的初始种子指令，即可自动生成高质量的监督微调（SFT）和直接偏好优化（DPO）数据 —— **全程无需任何人工标注**。

本框架基于单张 GPU 运行，使用 **DeepSeek API** 作为导师模型，并通过 **LLaMA-Factory** 训练轻量级的学生模型（**Qwen2.5-1.5B-Instruct**），从而摆脱了对多节点复杂硬件基础设施的依赖。

AutoIF 具备强大的**领域通用性**：用户只需更换种子指令文件，即可为法律、金融、医疗、教育等 30 多个垂直领域定制专属的微调模型。

> **设计原理：** AutoIF 将大模型的执行反馈转化为可扩展的对齐信号。在 SFT 阶段被过滤掉的负面样本，将转化为高对比度的 DPO 拒绝（Rejected）样本，从而最大化偏好学习所需的边距信号（Margin Signal）。

---

## 核心特性

- **跨领域适配流水线** — 只需更换一个种子指令文件即可适配任何专业领域，开箱即用支持 30 多个内置领域模板。
- **全自动化工作流** — 从原始种子指令到最终微调模型，全流程端到端自动执行，无需人工干预或审核。
- **单卡轻量化兼容** — 完整工作流可在单张 NVIDIA A800 (80 GB) 上平稳运行，大幅降低了硬件门槛。
- **执行验证确保数据质量** — 内置基于 Python 的自动化验证器，通过真实代码执行过滤训练样本，执行过滤器拒绝率达 ~71.7%，确保仅保留逻辑自洽且满足约束的样本。
- **两阶段精准对齐** — 结合 SFT（高分样本过滤，综合得分 > 9）与 DPO（构建选中/拒绝偏好对，通过率差异 > 0.5），实现对复杂约束的精准遵循。
- **灵活的提示词生成** — 同时支持 ShareGPT 风格的查询扩展以及导师模型模拟的响应生成。

---

## 技术栈

| 组件 | 技术方案 |
|---|---|
| **推理引擎** | vLLM 0.5.5 |
| **训练框架** | LLaMA-Factory |
| **导师（Teacher）模型** | DeepSeek API |
| **学生（Student）模型** | Qwen2.5-1.5B-Instruct (3 GB) |
| **NLI 过滤模型** | mDeBERTa-v3-base (2.5 GB) |
| **微调方法** | LoRA (SFT + DPO) |
| **计算精度** | BF16 |
| **运行环境** | Python 3.10+, CUDA 12.x, PyTorch 2.4.0 |

---

## 安装指南

### 环境要求

| 要求 | 技术规格 |
|---|---|
| **GPU 显存** | NVIDIA A800 (80 GB VRAM) 或同等规格 |
| **操作系统** | Ubuntu 20.04 / 22.04 |
| **Python 版本** | 3.10+ |
| **CUDA 版本** | 12.x |
| **核心依赖** | PyTorch 2.4.0, vLLM 0.5.5 (固定版本) |
| **磁盘空间** | ≥ 40 GB 剩余空间 |
| **API 密钥** | 数据合成阶段需要配置 DeepSeek API Key (导师模型) |

### 步骤 1 — 克隆项目并配置环境

```bash
bash scripts/setup.sh

```

该脚本会自动执行：

1. 安装支持 CUDA 12.1 的 PyTorch 2.4.0。
2. 安装用于推理加速的 vLLM 0.5.5。
3. 安装 LLaMA-Factory 作为后端训练框架。

### 步骤 2 — 下载学生模型与过滤模型

```bash
bash scripts/download_models.sh

```

下载学生模型（Qwen2.5-1.5B-Instruct）和自然语言推理（NLI）过滤模型（mDeBERTa-v3）。

### 步骤 3 — 配置 API 密钥

数据合成依赖于 DeepSeek API，请在运行流水线前配置您的环境变量：

```bash
export SUPERVISOR_API_KEY="YOUR_DEEPSEEK_API_KEY"

```

---

## 快速开始

### 选项 A：一键式完整流水线

使用编排脚本 `run_all.sh` 可以一键自动串联数据合成、SFT、DPO 以及 vLLM 部署测试，且原生支持垂直领域切换。

```bash
# 通用领域训练（默认）
bash scripts/run_all.sh

# 领域特定微调
bash scripts/run_all.sh --domain 法律
bash scripts/run_all.sh --domain 金融
bash scripts/run_all.sh --domain 医疗

# 在后台运行并实时监控日志
nohup bash scripts/run_all.sh --domain 法律 > run.log 2>&1 &
tail -f run.log

```

> **注意：** 该脚本会自动处理 LLaMA-Factory 的数据集注册，并在模型部署前自动应用 Qwen 专属的 `rope_scaling` 兼容性补丁。

#### 流水线阶段说明

| 阶段 | 描述 |
| --- | --- |
| **Stage 1** | AutoIF 9步 SFT 数据合成（通过 DeepSeek API） |
| **Stage 2** | DPO 偏好对数据构建 |
| **Stage 3** | 使用 LoRA 进行 SFT 监督微调训练 |
| **Stage 4** | SFT Stage 的 LoRA 权重合并 |
| **Stage 5** | 使用 LoRA 进行 DPO 偏好对齐训练 |
| **Stage 6** | DPO Stage 的 LoRA 权重合并 |
| **Stage 7** | 环境兼容性补丁修复 |
| **Stage 8** | 离线推理验证与 vLLM 部署测试 |

---

### 选项 B：分步手动执行

若您希望检查流水线内部的中间产物或进行定向调试，可以独立运行各个阶段的代码。

#### 阶段 1 — 数据合成 (AutoIF)

本阶段将调用 DeepSeek API，通过 9 步（含 RFT 增强、验证器函数生成、回译过滤等）构建 SFT 数据，并通过 3 步构建 DPO 数据。

```bash
# SFT 数据构建 (步骤 1–9)
python code_sft/1_RFT.py
# ... 顺序执行步骤 2 至步骤 8 ...
python code_sft/9_sft_data_construction.py

# DPO 数据构建
python code_dpo/1_dpo_rft_wash.py
python code_dpo/2_dpo_data_query_construct.py

```

#### 阶段 2 — SFT 微调与权重合并

通过 LLaMA-Factory 使用 LoRA 微调 Qwen2.5-1.5B。

```bash
cd LlamaFactory

# 启动 LoRA 微调
llamafactory-cli train ../configs/llamafactory_sft_lora.yaml

# 将 LoRA 权重合并至基座模型
llamafactory-cli export \
  --model_name_or_path ../models/student/Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ../models/model_d_sft \
  --export_dir ../models/model_d_sft_merged \
  --finetuning_type lora \
  --template qwen

cd ..

```

#### 阶段 3 — DPO 对齐与权重合并

在合并后的 SFT 模型基础之上，执行偏好强化学习对齐。

```bash
cd LlamaFactory

# 基于 SFT 合并模型进行 DPO 训练
llamafactory-cli train ../configs/llamafactory_dpo_lora.yaml

# 合并 DPO 权重（选择收敛效果最佳的 checkpoint-175）
llamafactory-cli export \
  --model_name_or_path ../models/model_d_sft_merged \
  --adapter_name_or_path ../models/model_d_dpo/checkpoint-175 \
  --export_dir ../models/model_d_dpo_merged \
  --finetuning_type lora \
  --template qwen

cd ..

```

#### 阶段 4 — 兼容性补丁与评估

应用相关修补脚本，解决 Qwen 模型权重合并后可能引发的 vLLM 兼容性报错。

```bash
python patches/fix_config.py
python patches/fix_qwen.py models/model_d_dpo_merged
python tests/models_to_test.py

```

#### 阶段 5 — vLLM 部署与 API 测试

使用高吞吐量推理引擎服务化最终的微调模型。

```bash
# 启动 vLLM 推理服务器
vllm serve models/model_d_dpo_merged \
  --dtype bfloat16 \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name qwen \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7

# 在另一个终端运行 API 测试脚本
python tests/test_vllm.py

```

<div align="center">
  <img src="images/test_vllm_result.png" width="800" alt="vLLM 推理测试结果">
</div>

---

## 领域适配

AutoIF 的核心创新在于：**只需更换种子指令文件，即可全自动生成高质量的特定垂直领域训练数据**。

### 内置领域列表 (30+)

| 领域大类 | 包含的垂直领域 |
| --- | --- |
| **基础科学** | 数学、物理、化学、生物、天文学、地理 |
| **工程技术** | 土木工程、机械工程、电气工程、化学工程、材料科学、能源工程 |
| **人文与社会科学** | 文学、历史、哲学、新闻学、社会学、心理学 |
| **商业与管理** | 工商管理、会计、公共管理、电子商务、金融 |
| **应用技术** | 法律、医学、教育、软件开发 |
| **艺术与体育** | 美术、音乐、体育 |

### 自定义领域配置

```bash
# 列出所有可用的内置领域模板
python scripts/generate_seed_instructions.py --list

# 借助 LLM 为新的自定义领域自动生成初始种子指令
python scripts/generate_seed_instructions.py --domain 建筑设计 --use-llm

# 为该自定义领域运行全自动微调流水线
bash scripts/run_all.sh --domain 建筑设计

```

> **贡献要求：** 若想贡献新的自定义领域模板，请将新条目添加至 `scripts/extended_domains.py` 中，并为该领域提供至少 **10 条具备代表性的初始种子指令**。

---

## 系统架构

### 数据合成流水线（9 步法）

| 步骤 | 操作说明 | 输入 → 输出样本变化 |
| --- | --- | --- |
| **第 1 步** | RFT 指令数据增强 | 36 个初始种子 → 830 条扩展指令 |
| **第 2 步** | 验证器函数（Validator Function）生成 | 830 + 36 = 866 条总指令 |
| **第 3 步** | 交叉验证过滤（Cross-Validation） | 866 → 491 条指令 (通过率: 56.7%) |
| **第 4 步** | 回译（Back-Translation）验证 | 491 条指令 |
| **第 5 步** | NLI 一致性过滤 | 491 → 426 条指令 (保留率: 86.76%) |
| **第 6 步** | 查询增强 + 响应数据批量生成 | 426 × ~10 提示词 × 5 响应 = 21,300 个候选样本 |
| **第 7 步** | 基于代码执行的质量评分 | 21,300 → 6,031 条 (通过真实 Python 执行) |
| **第 8 步** | 高质量 SFT 样本筛选 (得分 > 9) | 6,031 → 2,239 条高合规 SFT 样本 |
| **第 9 步** | SFT 数据集格式构建 | 最终输出文件: `IF_sft_data.json` |

<div align="center">
  <img src="images/评分段数据分布统计.png" width="700" alt="AutoIF 数据合成得分分布图">
</div>

### DPO 偏好对构建机制

为了保证负反馈信号的质量，DPO 流水线**特意绕过了**第 8 步的高分筛选机制，而是直接追溯回第 6 步的完整响应池。在 SFT 阶段因得分低而被丢弃的样本，在这里正好转化为高对比度的“拒绝（Rejected）”负例，从而在训练中提供极佳的偏好边界差异。

**DPO 步骤 1 (`1_dpo_rft_wash.py`):**

* 重新处理第 6 步产生的 4,260 个响应候选。
* 运行验证器函数计算每个响应的准确率得分。
* 输出格式为：`[response_text, accuracy_score]`。

**DPO 步骤 2 (`2_dpo_data_query_construct.py`):**

1. **正负样本分离：** 准确率得分 ≥ 0.5 的响应归为 `chosen`（选中）；得分 = 0 的响应归为 `rejected`（拒绝）。
2. **配对条件：** 一个有效的偏好对，必须在同一个提示词（Prompt）下同时拥有至少一个 `chosen` 和一个 `rejected` 响应。
3. **组合采样：** 每个提示词最多采样 2 个 `chosen` 和 2 个 `rejected` 响应，两两交叉组合生成所有有效的正负偏好对。

### 训练超参数设置

| 超参数名称 | SFT 微调阶段 | DPO 对齐阶段 |
| --- | --- | --- |
| **微调方法** | LoRA (rank=16, α=32) | LoRA (rank=16, α=32) |
| **学习率 (Learning Rate)** | 5e-5 | 5e-6 |
| **训练轮数 (Epochs)** | 3.0 | 2.0 |
| **最大序列长度** | 2048 | 2048 |
| **计算精度** | BF16 | BF16 |
| **批次大小 (单卡 / 梯度累积)** | 4 / 4 | 2 / 8 |
| **评估与保存频率** | 每 150 步一次 | 每 25 步一次 |
| **学习率调度器** | Cosine (warmup_ratio=0.05) | Cosine (warmup_ratio=0.1) |
| **LoRA 目标模块** | q, k, v, o, gate, up, down | q, k, v, o, gate, up, down |
| **DPO Beta 系数** | — | 0.3 |

---

## 训练指标

### SFT 收敛曲线

SFT 训练过程非常平稳，未出现过拟合迹象。验证损失（Validation Loss）从 1.41 丝滑下降至 1.20，并在大约 350 步时趋于稳定。

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

### DPO 偏好对齐

在 2 个 Epoch 的 DPO 偏好训练期间，奖励准确率（`Rewards/Accuracies`）稳步攀升并最终稳定在 **83%** 左右。评估损失在第 175 步达到最低点且未发生反弹；因此，系统最终选定 **`checkpoint-175`** 作为上产线的最终检查点。

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

---

## 评估结果

以下对比结果清晰地展现了基座模型、SFT 对齐模型以及 DPO 对齐模型在高难度指令遵循基准测试中的实际约束表现。

### 1. 基线（Baseline）：约束完全失效

在进行对齐训练前，原生的基座模型在单词推理中无法同时满足多项并发约束，格式、字符集以及词法边界约束全部失效。

<div align="center">
  <img src="images/base/base_response.png" width="750" alt="基座模型约束完全失效输出示例">
  <p><i>图：基座模型（Baseline）对多项并发约束完全失效的推理响应</i></p>
</div>

### 2. 渐进式对齐成功

**SFT 阶段（捕获基础约束）：** 模型成功掌握了目标格式约束（例如：输出全大写文本并正确添加 STOP 结束标记）。

<div align="center">
  <img src="images/SFT/SFT_response.png" width="750" alt="SFT阶段模型初步合规输出示例">
  <p><i>图：模型经过 SFT 微调后，已能初步捕获目标格式约束</i></p>
</div>

**DPO 阶段 — Checkpoint 175（完全化对齐）：** 通过基于偏好对的概率对齐，模型彻底内化了全部复合约束（满足电报体格式、严格限制在三句话内、所有句子均以 'B' 或 'T' 开头）。

<div align="center">
  <img src="images/DPO/DPO_response.png" width="750" alt="DPO阶段模型完全遵循约束输出示例">
  <p><i>图：Checkpoint 175 经过 DPO 偏好对齐后，完全 internalization 了所有复合约束</i></p>
</div>

---

## 流水线统计数据

以下数据源自 AutoDL 算力平台（单张 NVIDIA A800, 80 GB）运行通用领域微调基准测试的真实生产记录。

> **可重复性说明：** 由于大语言模型（LLM）的生成具有随机性，且 DPO 配对时采用了随机采样，因此独立多次运行后的绝对样本计数可能会有轻微浮动。

### 端到端数据流向统计

| 阶段 / 步骤 | 统计指标 | 样本计数 | 备注说明 |
| --- | --- | --- | --- |
| **第 0 步** | 原始种子指令数 | 36 | 基础初始种子集 |
| **第 1 步** | RFT 增强指令数 | 830 | 扩展后的总指令池 |
| **第 2 步** | 进入验证器构建的指令数 | 866 | 36 原始 + 830 增强 |
| **第 3 步** | 交叉验证存活数 | 491 | 过滤通过率: 56.7% |
| **第 5 步** | NLI 一致性过滤存活数 | 426 | 样本保留率: 86.76% |
| **第 6 步** | 总查询/响应候选样本数 | 21,300 | 426 核心指令 × ~10 提示词 × 5 响应 |
| **第 7 步** | 执行验证通过数 | 6,031 | 成功通过 Python 执行器的样本（Accuracy > 0） |
| **第 8 步** | 高质量 SFT 训练样本数 | 2,239 | 综合评分 > 9 / 10 的核心样本 |
| **DPO** | 最终 DPO 偏好对数量 | 2,159 | 完美满足选中/拒绝得分阈值（差异 ≥ 0.5） |

### 执行过滤器拒绝率 (针对第 7 步)

在第 6 步生成的 21,300 个候选响应样本中：

* **通过 Python 真实执行验证：** 6,031 个样本
* **被 Python 执行验证器拦截拒绝：** 15,269 个样本
* **执行过滤器整体拒绝率高达：71.7%**

---

## 项目结构

```text
AutoIF-LLM/
├── .gitignore
├── README.md
├── requirements.txt               # 固定的 Python 依赖包列表
├── code_dpo/                      # DPO 偏好对构建流水线代码
│   ├── 1_dpo_rft_wash.py
│   └── 2_dpo_data_query_construct.py
├── code_sft/                      # AutoIF SFT 数据合成流水线代码
│   ├── 1_RFT.py
│   ├── ...
│   ├── 9_sft_data_construction.py
│   └── utils.py                   # 核心环境变量与配置常量
├── configs/                       # LLaMA-Factory 训练与数据集配置文件
│   ├── llamafactory_sft_lora.yaml
│   ├── llamafactory_dpo_lora.yaml
│   ├── llama_factory_dataset_info.json
│   └── pipeline_config.yaml
├── images/                        # 文档图表与评估结果图像资产
│   ├── base/
│   ├── DPO/
│   └── SFT/
├── output/                        # 运行期间的临时数据与最终数据集输出目录
│   ├── dpo_pairs_flat.jsonl
│   └── IF_sft_data.json
├── patches/                       # 上游依赖环境兼容性补丁
│   ├── dpo2_patches.py
│   ├── fix_config.py
│   └── fix_qwen.py
├── sample_data/
│   └── seed_instruction.txt       # 默认的初始种子指令文件
├── scripts/                       # 自动化基础设施与环境初始化脚本
│   ├── download_models.sh
│   ├── extended_domains.py
│   ├── generate_seed_instructions.py
│   ├── run_all.sh
│   └── setup.sh
├── tests/                         # 推理验证与模型部署测试脚本
│   ├── models_to_test.py
│   └── test_vllm.py
└── tools/
    └── view_scores.py             # 数据质量得分可视化分析工具

```

---

## 故障排除

由于上游开源依赖库更新过于频繁，环境可能偶尔出现冲突。AutoIF 在 `run_all.sh` 脚本中内置了三项兼容性补丁，会在流水线运行时自动应用。

| 补丁脚本 | 触发条件 | 解决方案 | 手动执行命令 |
| --- | --- | --- | --- |
| **`fix_qwen.py`** | vLLM 启动时报 `rope_scaling` 校验错误并崩溃 | 自动向模型的 `config.json` 中注入 `{"factor": 1.0, "type": "default"}` 字段 | `python patches/fix_qwen.py ./models/model_d_dpo_merged` |
| **`fix_config.py`** | 训练框架无法解析非标准的某些位置编码（Positional Encoding）字段 | 自动遍历 `models/` 下的模型配置并剔除不兼容的参数 | `python patches/fix_config.py` |
| **`dpo2_patches.py`** | 嵌套的 ShareGPT 对话数组结构在微调框架中引发解析异常 | 将 DPO 数据平铺展平为更稳定的 Alpaca 格式并重新注册数据集 | `python patches/dpo2_patches.py` |

---

## 贡献指南

非常欢迎社区提交贡献！请遵循标准的 GitHub 开发工作流：

1. **Fork** 本仓库。
2. 创建您的特性分支：`git checkout -b feature/your-feature-name`
3. 提交更改，并编写清晰、具备描述性的 Commit 信息。
4. 提交 **Pull Request** 指向本仓库的 `main` 分支。

若是贡献新的行业领域模板，请将对应的领域信息添加至 `scripts/extended_domains.py` 中，并附带至少 **10 条代表性种子指令**。

---

## 论文引用

如果您在学术研究中使用了 AutoIF 框架，或者本项目对您的工作有所启发，请引用原作者的论文：

```bibtex
@article{dong2024self,
  title={Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models},
  author={Dong, Guanting and Lu, Keming and Li, Chengpeng and Xia, Tingyu and Yu, Bowen and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2406.13542},
  year={2024}
}

```

---

## 开源许可证

本项目基于 **Apache License 2.0** 许可证开源。

项目中所依赖的下游基础模型（如 Qwen2.5 系列、mDeBERTa-v3 等）受其各自原生开源许可证的约束。在进行任何商业化应用前，请务必仔细阅读并遵循相关模型的开源协议。

```

```没问题！这里已经为你整理好了完全中文化的 `README.md` 源码。我不仅把文字内容翻译成了流畅、专业的中文技术文档术语，还顺手**彻底清理掉了原本夹杂在中间的 `` 乱码标记**，并保留了所有排版和跳转链接。

你可以直接点击代码块右上角的 **“复制”** 按钮，直接粘贴到你的 GitHub 项目中：

```markdown
<div align="center">

# AutoIF-LLM

**基于执行反馈与自我博弈的自动化指令遵循对齐框架**

*基于论文：[Self-play with Execution Feedback (ICLR 2025 Spotlight)](https://arxiv.org/abs/2406.13542)*

---

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Student%20Model-Qwen2.5--1.5B-purple)](https://huggingface.co/Qwen)
[![Powered by](https://img.shields.io/badge/Teacher%20Model-DeepSeek%20API-00BFFF)](https://platform.deepseek.com/)

</div>

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [领域适配](#领域适配)
- [系统架构](#系统架构)
- [训练指标](#训练指标)
- [评估结果](#评估结果)
- [流水线统计数据](#流水线统计数据)
- [项目结构](#项目结构)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [论文引用](#论文引用)
- [开源许可证](#开源许可证)

---

## 项目概述

**AutoIF-LLM** 是一个全自动化的微调框架，旨在通过**执行反馈（Execution Feedback）**和**自我博弈（Self-Play）**来显著提升大语言模型（LLM）的指令遵循能力。该框架采用“导师-学生（Teacher-Student）”架构以及多阶段数据合成流水线，仅需极少量的初始种子指令，即可自动生成高质量的监督微调（SFT）和直接偏好优化（DPO）数据 —— **全程无需任何人工标注**。

本框架基于单张 GPU 运行，使用 **DeepSeek API** 作为导师模型，并通过 **LLaMA-Factory** 训练轻量级的学生模型（**Qwen2.5-1.5B-Instruct**），从而摆脱了对多节点复杂硬件基础设施的依赖。

AutoIF 具备强大的**领域通用性**：用户只需更换种子指令文件，即可为法律、金融、医疗、教育等 30 多个垂直领域定制专属的微调模型。

> **设计原理：** AutoIF 将大模型的执行反馈转化为可扩展的对齐信号。在 SFT 阶段被过滤掉的负面样本，将转化为高对比度的 DPO 拒绝（Rejected）样本，从而最大化偏好学习所需的边距信号（Margin Signal）。

---

## 核心特性

- **跨领域适配流水线** — 只需更换一个种子指令文件即可适配任何专业领域，开箱即用支持 30 多个内置领域模板。
- **全自动化工作流** — 从原始种子指令到最终微调模型，全流程端到端自动执行，无需人工干预或审核。
- **单卡轻量化兼容** — 完整工作流可在单张 NVIDIA A800 (80 GB) 上平稳运行，大幅降低了硬件门槛。
- **执行验证确保数据质量** — 内置基于 Python 的自动化验证器，通过真实代码执行过滤训练样本，执行过滤器拒绝率达 ~71.7%，确保仅保留逻辑自洽且满足约束的样本。
- **两阶段精准对齐** — 结合 SFT（高分样本过滤，综合得分 > 9）与 DPO（构建选中/拒绝偏好对，通过率差异 > 0.5），实现对复杂约束的精准遵循。
- **灵活的提示词生成** — 同时支持 ShareGPT 风格的查询扩展以及导师模型模拟的响应生成。

---

## 技术栈

| 组件 | 技术方案 |
|---|---|
| **推理引擎** | vLLM 0.5.5 |
| **训练框架** | LLaMA-Factory |
| **导师（Teacher）模型** | DeepSeek API |
| **学生（Student）模型** | Qwen2.5-1.5B-Instruct (3 GB) |
| **NLI 过滤模型** | mDeBERTa-v3-base (2.5 GB) |
| **微调方法** | LoRA (SFT + DPO) |
| **计算精度** | BF16 |
| **运行环境** | Python 3.10+, CUDA 12.x, PyTorch 2.4.0 |

---

## 安装指南

### 环境要求

| 要求 | 技术规格 |
|---|---|
| **GPU 显存** | NVIDIA A800 (80 GB VRAM) 或同等规格 |
| **操作系统** | Ubuntu 20.04 / 22.04 |
| **Python 版本** | 3.10+ |
| **CUDA 版本** | 12.x |
| **核心依赖** | PyTorch 2.4.0, vLLM 0.5.5 (固定版本) |
| **磁盘空间** | ≥ 40 GB 剩余空间 |
| **API 密钥** | 数据合成阶段需要配置 DeepSeek API Key (导师模型) |

### 步骤 1 — 克隆项目并配置环境

```bash
bash scripts/setup.sh

```

该脚本会自动执行：

1. 安装支持 CUDA 12.1 的 PyTorch 2.4.0。
2. 安装用于推理加速的 vLLM 0.5.5。
3. 安装 LLaMA-Factory 作为后端训练框架。

### 步骤 2 — 下载学生模型与过滤模型

```bash
bash scripts/download_models.sh

```

下载学生模型（Qwen2.5-1.5B-Instruct）和自然语言推理（NLI）过滤模型（mDeBERTa-v3）。

### 步骤 3 — 配置 API 密钥

数据合成依赖于 DeepSeek API，请在运行流水线前配置您的环境变量：

```bash
export SUPERVISOR_API_KEY="YOUR_DEEPSEEK_API_KEY"

```

---

## 快速开始

### 选项 A：一键式完整流水线

使用编排脚本 `run_all.sh` 可以一键自动串联数据合成、SFT、DPO 以及 vLLM 部署测试，且原生支持垂直领域切换。

```bash
# 通用领域训练（默认）
bash scripts/run_all.sh

# 领域特定微调
bash scripts/run_all.sh --domain 法律
bash scripts/run_all.sh --domain 金融
bash scripts/run_all.sh --domain 医疗

# 在后台运行并实时监控日志
nohup bash scripts/run_all.sh --domain 法律 > run.log 2>&1 &
tail -f run.log

```

> **注意：** 该脚本会自动处理 LLaMA-Factory 的数据集注册，并在模型部署前自动应用 Qwen 专属的 `rope_scaling` 兼容性补丁。

#### 流水线阶段说明

| 阶段 | 描述 |
| --- | --- |
| **Stage 1** | AutoIF 9步 SFT 数据合成（通过 DeepSeek API） |
| **Stage 2** | DPO 偏好对数据构建 |
| **Stage 3** | 使用 LoRA 进行 SFT 监督微调训练 |
| **Stage 4** | SFT Stage 的 LoRA 权重合并 |
| **Stage 5** | 使用 LoRA 进行 DPO 偏好对齐训练 |
| **Stage 6** | DPO Stage 的 LoRA 权重合并 |
| **Stage 7** | 环境兼容性补丁修复 |
| **Stage 8** | 离线推理验证与 vLLM 部署测试 |

---

### 选项 B：分步手动执行

若您希望检查流水线内部的中间产物或进行定向调试，可以独立运行各个阶段的代码。

#### 阶段 1 — 数据合成 (AutoIF)

本阶段将调用 DeepSeek API，通过 9 步（含 RFT 增强、验证器函数生成、回译过滤等）构建 SFT 数据，并通过 3 步构建 DPO 数据。

```bash
# SFT 数据构建 (步骤 1–9)
python code_sft/1_RFT.py
# ... 顺序执行步骤 2 至步骤 8 ...
python code_sft/9_sft_data_construction.py

# DPO 数据构建
python code_dpo/1_dpo_rft_wash.py
python code_dpo/2_dpo_data_query_construct.py

```

#### 阶段 2 — SFT 微调与权重合并

通过 LLaMA-Factory 使用 LoRA 微调 Qwen2.5-1.5B。

```bash
cd LlamaFactory

# 启动 LoRA 微调
llamafactory-cli train ../configs/llamafactory_sft_lora.yaml

# 将 LoRA 权重合并至基座模型
llamafactory-cli export \
  --model_name_or_path ../models/student/Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ../models/model_d_sft \
  --export_dir ../models/model_d_sft_merged \
  --finetuning_type lora \
  --template qwen

cd ..

```

#### 阶段 3 — DPO 对齐与权重合并

在合并后的 SFT 模型基础之上，执行偏好强化学习对齐。

```bash
cd LlamaFactory

# 基于 SFT 合并模型进行 DPO 训练
llamafactory-cli train ../configs/llamafactory_dpo_lora.yaml

# 合并 DPO 权重（选择收敛效果最佳的 checkpoint-175）
llamafactory-cli export \
  --model_name_or_path ../models/model_d_sft_merged \
  --adapter_name_or_path ../models/model_d_dpo/checkpoint-175 \
  --export_dir ../models/model_d_dpo_merged \
  --finetuning_type lora \
  --template qwen

cd ..

```

#### 阶段 4 — 兼容性补丁与评估

应用相关修补脚本，解决 Qwen 模型权重合并后可能引发的 vLLM 兼容性报错。

```bash
python patches/fix_config.py
python patches/fix_qwen.py models/model_d_dpo_merged
python tests/models_to_test.py

```

#### 阶段 5 — vLLM 部署与 API 测试

使用高吞吐量推理引擎服务化最终的微调模型。

```bash
# 启动 vLLM 推理服务器
vllm serve models/model_d_dpo_merged \
  --dtype bfloat16 \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name qwen \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7

# 在另一个终端运行 API 测试脚本
python tests/test_vllm.py

```

> **运行截图占位符** — `images/test_vllm_result.png`

---

## 领域适配

AutoIF 的核心创新在于：**只需更换种子指令文件，即可全自动生成高质量的特定垂直领域训练数据**。

### 内置领域列表 (30+)

| 领域大类 | 包含的垂直领域 |
| --- | --- |
| **基础科学** | 数学、物理、化学、生物、天文学、地理 |
| **工程技术** | 土木工程、机械工程、电气工程、化学工程、材料科学、能源工程 |
| **人文与社会科学** | 文学、历史、哲学、新闻学、社会学、心理学 |
| **商业与管理** | 工商管理、会计、公共管理、电子商务、金融 |
| **应用技术** | 法律、医学、教育、软件开发 |
| **艺术与体育** | 美术、音乐、体育 |

### 自定义领域配置

```bash
# 列出所有可用的内置领域模板
python scripts/generate_seed_instructions.py --list

# 借助 LLM 为新的自定义领域自动生成初始种子指令
python scripts/generate_seed_instructions.py --domain 建筑设计 --use-llm

# 为该自定义领域运行全自动微调流水线
bash scripts/run_all.sh --domain 建筑设计

```

> **贡献要求：** 若想贡献新的自定义领域模板，请将新条目添加至 `scripts/extended_domains.py` 中，并为该领域提供至少 **10 条具备代表性的初始种子指令**。

---

## 系统架构

### 数据合成流水线（9 步法）

| 步骤 | 操作说明 | 输入 → 输出样本变化 |
| --- | --- | --- |
| **第 1 步** | RFT 指令数据增强 | 36 个初始种子 → 830 条扩展指令 |
| **第 2 步** | 验证器函数（Validator Function）生成 | 830 + 36 = 866 条总指令 |
| **第 3 步** | 交叉验证过滤（Cross-Validation） | 866 → 491 条指令 (通过率: 56.7%) |
| **第 4 步** | 回译（Back-Translation）验证 | 491 条指令 |
| **第 5 步** | NLI 一致性过滤 | 491 → 426 条指令 (保留率: 86.76%) |
| **第 6 步** | 查询增强 + 响应数据批量生成 | 426 × ~10 提示词 × 5 响应 = 21,300 个候选样本 |
| **第 7 步** | 基于代码执行的质量评分 | 21,300 → 6,031 条 (通过真实 Python 执行) |
| **第 8 步** | 高质量 SFT 样本筛选 (得分 > 9) | 6,031 → 2,239 条高合规 SFT 样本 |
| **第 9 步** | SFT 数据集格式构建 | 最终输出文件: `IF_sft_data.json` |

> **运行截图占位符** — 得分分布可视化图表：`images/score_distribution.png`

### DPO 偏好对构建机制

为了保证负反馈信号的质量，DPO 流水线**特意绕过了**第 8 步的高分筛选机制，而是直接追溯回第 6 步的完整响应池。在 SFT 阶段因得分低而被丢弃的样本，在这里正好转化为高对比度的“拒绝（Rejected）”负例，从而在训练中提供极佳的偏好边界差异。

**DPO 步骤 1 (`1_dpo_rft_wash.py`):**

* 重新处理第 6 步产生的 4,260 个响应候选。
* 运行验证器函数计算每个响应的准确率得分。
* 输出格式为：`[response_text, accuracy_score]`。

**DPO 步骤 2 (`2_dpo_data_query_construct.py`):**

1. **正负样本分离：** 准确率得分 ≥ 0.5 的响应归为 `chosen`（选中）；得分 = 0 的响应归为 `rejected`（拒绝）。
2. **配对条件：** 一个有效的偏好对，必须在同一个提示词（Prompt）下同时拥有至少一个 `chosen` 和一个 `rejected` 响应。
3. **组合采样：** 每个提示词最多采样 2 个 `chosen` 和 2 个 `rejected` 响应，两两交叉组合生成所有有效的正负偏好对。

### 训练超参数设置

| 超参数名称 | SFT 微调阶段 | DPO 对齐阶段 |
| --- | --- | --- |
| **微调方法** | LoRA (rank=16, α=32) | LoRA (rank=16, α=32) |
| **学习率 (Learning Rate)** | 5e-5 | 5e-6 |
| **训练轮数 (Epochs)** | 3.0 | 2.0 |
| **最大序列长度** | 2048 | 2048 |
| **计算精度** | BF16 | BF16 |
| **批次大小 (单卡 / 梯度累积)** | 4 / 4 | 2 / 8 |
| **评估与保存频率** | 每 150 步一次 | 每 25 步一次 |
| **学习率调度器** | Cosine (warmup_ratio=0.05) | Cosine (warmup_ratio=0.1) |
| **LoRA 目标模块** | q, k, v, o, gate, up, down | q, k, v, o, gate, up, down |
| **DPO Beta 系数** | — | 0.3 |

---

## 训练指标

### SFT 收敛曲线

SFT 训练过程非常平稳，未出现过拟合迹象。验证损失（Validation Loss）从 1.41 丝滑下降至 1.20，并在大约 350 步时趋于稳定。

> **运行截图占位符** — SFT 训练损失: `images/SFT/training_loss.png`
> **运行截图占位符** — SFT 验证损失: `images/SFT/training_eval_loss.png`

### DPO 偏好对齐

在 2 个 Epoch 的 DPO 偏好训练期间，奖励准确率（`Rewards/Accuracies`）稳步攀升并最终稳定在 **83%** 左右。评估损失在第 175 步达到最低点且未发生反弹；因此，系统最终选定 **`checkpoint-175`** 作为上产线的最终检查点。

> **运行截图占位符** — DPO 奖励准确率: `images/DPO/dpo_training_rewards_accuracies.png`
> **运行截图占位符** — DPO 评估损失: `images/DPO/dpo_training_eval_loss.png`
> **运行截图占位符** — DPO 训练损失: `images/DPO/dpo_training_loss.png`

---

## 评估结果

以下对比结果清晰地展现了基座模型、SFT 对齐模型以及 DPO 对齐模型在高难度指令遵循基准测试中的实际约束表现。

### 1. 基线（Baseline）：约束完全失效

在进行对齐训练前，原生的基座模型在单词推理中无法同时满足多项并发约束，格式、字符集以及词法边界约束全部失效。

> **运行截图占位符** — `images/base/base_response.png`

### 2. 渐进式对齐成功

**SFT 阶段（捕获基础约束）：** 模型成功掌握了目标格式约束（例如：输出全大写文本并正确添加 STOP 结束标记）。

> **运行截图占位符** — `images/SFT/SFT_response.png`

**DPO 阶段 — Checkpoint 175（完全化对齐）：** 通过基于偏好对的概率对齐，模型彻底内化了全部复合约束（满足电报体格式、严格限制在三句话内、所有句子均以 'B' 或 'T' 开头）。

> **运行截图占位符** — `images/DPO/DPO_response.png`

---

## 流水线统计数据

以下数据源自 AutoDL 算力平台（单张 NVIDIA A800, 80 GB）运行通用领域微调基准测试的真实生产记录。

> **可重复性说明：** 由于大语言模型（LLM）的生成具有随机性，且 DPO 配对时采用了随机采样，因此独立多次运行后的绝对样本计数可能会有轻微浮动。

### 端到端数据流向统计

| 阶段 / 步骤 | 统计指标 | 样本计数 | 备注说明 |
| --- | --- | --- | --- |
| **第 0 步** | 原始种子指令数 | 36 | 基础初始种子集 |
| **第 1 步** | RFT 增强指令数 | 830 | 扩展后的总指令池 |
| **第 2 步** | 进入验证器构建的指令数 | 866 | 36 原始 + 830 增强 |
| **第 3 步** | 交叉验证存活数 | 491 | 过滤通过率: 56.7% |
| **第 5 步** | NLI 一致性过滤存活数 | 426 | 样本保留率: 86.76% |
| **第 6 步** | 总查询/响应候选样本数 | 21,300 | 426 核心指令 × ~10 提示词 × 5 响应 |
| **第 7 步** | 执行验证通过数 | 6,031 | 成功通过 Python 执行器的样本（Accuracy > 0） |
| **第 8 步** | 高质量 SFT 训练样本数 | 2,239 | 综合评分 > 9 / 10 的核心样本 |
| **DPO** | 最终 DPO 偏好对数量 | 2,159 | 完美满足选中/拒绝得分阈值（差异 ≥ 0.5） |

### 执行过滤器拒绝率 (针对第 7 步)

在第 6 步生成的 21,300 个候选响应样本中：

* **通过 Python 真实执行验证：** 6,031 个样本
* **被 Python 执行验证器拦截拒绝：** 15,269 个样本
* **执行过滤器整体拒绝率高达：71.7%**

---

## 项目结构

```text
AutoIF-LLM/
├── .gitignore
├── README.md
├── requirements.txt               # 固定的 Python 依赖包列表
├── code_dpo/                      # DPO 偏好对构建流水线代码
│   ├── 1_dpo_rft_wash.py
│   └── 2_dpo_data_query_construct.py
├── code_sft/                      # AutoIF SFT 数据合成流水线代码
│   ├── 1_RFT.py
│   ├── ...
│   ├── 9_sft_data_construction.py
│   └── utils.py                   # 核心环境变量与配置常量
├── configs/                       # LLaMA-Factory 训练与数据集配置文件
│   ├── llamafactory_sft_lora.yaml
│   ├── llamafactory_dpo_lora.yaml
│   ├── llama_factory_dataset_info.json
│   └── pipeline_config.yaml
├── images/                        # 文档图表与评估结果图像资产
│   ├── base/
│   ├── DPO/
│   └── SFT/
├── output/                        # 运行期间的临时数据与最终数据集输出目录
│   ├── dpo_pairs_flat.jsonl
│   └── IF_sft_data.json
├── patches/                       # 上游依赖环境兼容性补丁
│   ├── dpo2_patches.py
│   ├── fix_config.py
│   └── fix_qwen.py
├── sample_data/
│   └── seed_instruction.txt       # 默认的初始种子指令文件
├── scripts/                       # 自动化基础设施与环境初始化脚本
│   ├── download_models.sh
│   ├── extended_domains.py
│   ├── generate_seed_instructions.py
│   ├── run_all.sh
│   └── setup.sh
├── tests/                         # 推理验证与模型部署测试脚本
│   ├── models_to_test.py
│   └── test_vllm.py
└── tools/
    └── view_scores.py             # 数据质量得分可视化分析工具

```

---

## 故障排除

由于上游开源依赖库更新过于频繁，环境可能偶尔出现冲突。AutoIF 在 `run_all.sh` 脚本中内置了三项兼容性补丁，会在流水线运行时自动应用。

| 补丁脚本 | 触发条件 | 解决方案 | 手动执行命令 |
| --- | --- | --- | --- |
| **`fix_qwen.py`** | vLLM 启动时报 `rope_scaling` 校验错误并崩溃 | 自动向模型的 `config.json` 中注入 `{"factor": 1.0, "type": "default"}` 字段 | `python patches/fix_qwen.py ./models/model_d_dpo_merged` |
| **`fix_config.py`** | 训练框架无法解析非标准的某些位置编码（Positional Encoding）字段 | 自动遍历 `models/` 下的模型配置并剔除不兼容的参数 | `python patches/fix_config.py` |
| **`dpo2_patches.py`** | 嵌套的 ShareGPT 对话数组结构在微调框架中引发解析异常 | 将 DPO 数据平铺展平为更稳定的 Alpaca 格式并重新注册数据集 | `python patches/dpo2_patches.py` |

---

## 贡献指南

非常欢迎社区提交贡献！请遵循标准的 GitHub 开发工作流：

1. **Fork** 本仓库。
2. 创建您的特性分支：`git checkout -b feature/your-feature-name`
3. 提交更改，并编写清晰、具备描述性的 Commit 信息。
4. 提交 **Pull Request** 指向本仓库的 `main` 分支。

若是贡献新的行业领域模板，请将对应的领域信息添加至 `scripts/extended_domains.py` 中，并附带至少 **10 条代表性种子指令**。

---

## 论文引用

如果您在学术研究中使用了 AutoIF 框架，或者本项目对您的工作有所启发，请引用原作者的论文：

```bibtex
@article{dong2024self,
  title={Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models},
  author={Dong, Guanting and Lu, Keming and Li, Chengpeng and Xia, Tingyu and Yu, Bowen and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2406.13542},
  year={2024}
}

```

---

## 开源许可证

本项目基于 **Apache License 2.0** 许可证开源。

项目中所依赖的下游基础模型（如 Qwen2.5 系列、mDeBERTa-v3 等）受其各自原生开源许可证的约束。在进行任何商业化应用前，请务必仔细阅读并遵循相关模型的开源协议。

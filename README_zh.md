<div align="center">

# AutoIF
### 大语言模型指令遵循全自动微调框架
*基于执行反馈的自博弈机制 (ICLR 2025 Spotlight)*

</div>

---

## 目录
- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [领域自适应](#领域自适应)
- [系统架构](#系统架构)
- [评测结果](#评测结果)
- [流水线统计数据](#流水线统计数据)
- [项目结构](#项目结构)
- [故障排除与补丁](#故障排除与补丁)
- [参与贡献](#参与贡献)
- [论文引用](#论文引用)
- [开源协议](#开源协议)

---

## 项目概述
AutoIF 是一个全自动的微调框架，旨在通过**执行反馈**（Execution Feedback）和**自博弈机制**（Self-play）显著提升大语言模型（LLM）的指令遵循能力。通过构建“教师-学生”双模型架构和多阶段数据合成流水线，AutoIF 仅需输入初始种子指令，即可全自动生成高质量的监督微调（SFT）和直接偏好优化（DPO）训练数据，整个流程无需任何人工标注。

本框架具备极强的**领域通用性**：用户只需更换种子指令文件，即可在单张 GPU 上、约 20 分钟内针对法律、金融、医疗、教育等 30 多个专业领域全自动提炼并训练出领域专属大模型。

> **核心设计理念：** AutoIF 将大模型自身的执行反馈转化为可扩展的对齐信号，巧妙地将模型生成的“错误/负面输出”转化为 DPO 偏好学习中极具价值的负样本材料。

---

## 核心特性
* **领域通用流水线** — 仅需更换单个种子指令文件即可快速迁移至任何专业领域，系统开箱即用自带 30 多个内置领域模板。
* **全自动化端到端** — 从原始种子指令输入，到数据合成、模型微调、权重合并以及最后的量化，全流程一键执行，无需人工干预或审核。
* **单卡高效兼容** — 完整工作流仅需单张 NVIDIA A800 (80 GB) 显卡即可完美跑通，无需复杂的分布式多机多卡基础设施。
* **超高迭代速度** — 剔除基础环境配置时间，核心数据合成与双阶段对齐微调流水线在 20 分钟内即可全部完成。
* **基于执行验证的数据质量** — 内置基于 Python 解释器的自动化验证器，通过真实代码执行严格过滤候选样本。实验表明该步骤拦截率约为 66.6%，从根本上确保了留存样本的语义连贯性与约束满足度。
* **双阶段对齐优化** — 结合了 SFT 阶段（高分样本筛选）和 DPO 阶段（基于通过/失败切分的偏好对构建），双管齐下最大化模型的约束遵循能力。

---

## 技术栈

| 组件 | 核心技术 |
| :--- | :--- |
| **推理加速引擎** | vLLM 0.5.5 |
| **微调训练框架** | LLaMA-Factory |
| **教师模型 (Teacher)** | Qwen2.5-7B-Instruct (15 GB) |
| **学生模型 (Student)** | Qwen2.5-1.5B (3 GB) |
| **自然语言推理过滤模型** | mDeBERTa-v3-base (2.5 GB) |
| **微调训练算法** | LoRA (SFT + DPO) |
| **计算精度** | BF16 |
| **运行环境** | Python 3.10+, CUDA 12.x, PyTorch 2.4.0 |

---

## 环境安装

### 前提条件
| 硬件/软件要求 | 具体规格 |
| :--- | :--- |
| **显卡 (GPU)** | NVIDIA A800 (80 GB VRAM) |
| **操作系统** | Ubuntu 20.04 / 22.04 |
| **Python 版本** | 3.10 及以上 |
| **CUDA 版本** | 12.x |
| **磁盘空间** | 剩余可用空间 $\ge$ 40 GB |

### 步骤 1 — 上传并解压项目文件
```bash
cd /root/autodl-tmp
unzip AutoIF-LLM.zip && cd AutoIF-LLM

```

### 步骤 2 — 运行一键环境配置脚本

```bash
bash setup.sh

```

该脚本会自动执行以下操作：

1. 安装支持 CUDA 12.1 的 PyTorch 2.4.0。
2. 安装用于高性能推理加速的 vLLM 0.5.5。
3. 安装大模型微调后端 LLaMA-Factory。
4. 自动下载流水线所需的上游模型权重（Qwen2.5-7B-Instruct, Qwen2.5-1.5B 以及 mDeBERTa-v3）。

---

## 快速开始

### 运行全自动流水线

```bash
# 运行通用领域微调（默认配置）
bash run_all.sh

# 运行特定专业领域的自适应微调
bash run_all.sh --domain 法律      # 法律领域
bash run_all.sh --domain 金融      # 金融领域
bash run_all.sh --domain 医疗      # 医疗领域

# 后台静默运行并实时监控日志
nohup bash run_all.sh --domain 法律 > run.log 2>&1 &
tail -f run.log

```

### 流水线阶段一览

`run_all.sh` 脚本会严格按照以下阶段顺序串联执行：

| 阶段 | 阶段名称与描述 |
| --- | --- |
| **Stage 1** | 启动 vLLM 教师模型推理服务器 |
| **Stage 2** | 执行 AutoIF 核心的 9 步数据合成流水线 |
| **Stage 3** | 构建 DPO 偏好对（Preference Pairs）数据集 |
| **Stage 4** | 基于 LoRA 算法进行第一阶段的 SFT 监督微调训练 |
| **Stage 5** | 基于 LoRA 算法进行第二阶段的 DPO 偏好对齐训练 |
| **Stage 6** | 自动合并 LoRA 权重与基座模型 |
| **Stage 7** | 执行模型量化导出 |
| **Stage 8** | 运行最终微调模型的推理基准验证 |

---

## 领域自适应

AutoIF 的核心创新在于通过直接替换**种子指令**，即可工业化、规模化地自动合成高质量特定领域数据。

```text
种子指令输入 → AutoIF 流水线提炼 → SFT 数据 + DPO 数据 → LoRA 阶段微调 → 领域专属定制 LLM
      ↑                                                                                ↓
  一键快速替换                                                                     覆盖任意垂直领域

```

### 已内置支持的专业领域 (30+)

* **基础科学:** 数学、物理、化学、生物、天文学、地理学
* **工程技术:** 土木工程、机械工程、电气工程、化学工程、材料科学、能源工程
* **人文社科:** 文学、历史学、哲学、新闻学、社会学、心理学
* **商业管理:** 工商管理、会计学、公共管理、电子商务、金融学
* **应用垂直领域:** 法律、医学、教育学、计算机编程
* **艺术与体育:** 美术、音乐、体育运动

### 自定义垂直领域配置命令

```bash
# 列出系统当前所有内置的专业领域
python scripts/generate_seed_instructions.py --list

# 借助大模型为全新的自定义垂直领域自动生成初始种子指令
python scripts/generate_seed_instructions.py --domain 建筑设计 --use-llm

# 使用刚生成的自定义领域数据一键启动完整流水线
bash run_all.sh --domain 建筑设计

```

---

## 系统架构

### 9 步数据合成流水线 (Data Synthesis Pipeline)

* **Step 1:** 指令扩充阶段（通过 RFT 机制将初始 36 条种子指令扩充至 189 条）
* **Step 2:** 验证函数自动生成阶段（为每条指令自动编写 Python 校验代码）
* **Step 3:** 交叉验证与严格过滤（189 条筛选保留 70 条高可靠验证指令）
* **Step 4:** 语义反向翻译阶段（Back-Translation）
* **Step 5:** 基于 NLI（自然语言推理）的一致性过滤（70 条留存 59 条，留存率 86.76%）
* **Step 6:** Query 扩充与多候选响应生成（16 倍 Query 扩充 $\times$ 5 个响应 $\rightarrow$ 4,720 个候选样本）
* **Step 7:** 基于代码执行的样本质量动态评分（获得 1,578 个通过真实执行的有效样本）
* **Step 8:** 高质量微调过滤（筛选评分 Score > 8 的高分样本 $\rightarrow$ 27 个极精炼 SFT 样本）
* **Step 9:** SFT 数据集最终构建（导出至 `IF_sft_data.json`）

### DPO 偏好对构建逻辑

```text
Step 7 候选响应全量池 (共 4,720 个样本)
    ├── Chosen 胜出组  (准确率 accuracy ≥ 0.7)  ──┐
    └── Rejected 淘汰组 (准确率 accuracy = 0.0) ──┴─→ 笛卡尔积交叉配对 → 587 组标准 DPO 偏好对
                                                                     (dpo_pairs_flat.jsonl)

```

> **架构设计考量：** DPO 偏好对的构建刻意绕过了 Step 8 的高分筛选器，直接回溯到 Step 6/7 的全量响应池。在 SFT 阶段被过滤掉的 3,142 个“失败执行样本”，在 DPO 阶段恰恰形成了天然、高反差的负面教材（Negative Examples），能够最大化偏好对齐算法所需的 Preference Margin。

### 模型训练超参数配置

| 超参数名称 | 监督微调阶段 (SFT Phase) | 偏好对齐阶段 (DPO Phase) |
| --- | --- | --- |
| **微调算法** | LoRA (rank=32, $\alpha$=64) | LoRA (rank=8, $\alpha$=16) |
| **初始学习率 (Learning Rate)** | 1e-4 | 3e-6 |
| **训练轮数 (Epochs)** | 15 | 5 |
| **最大序列长度 (Max Length)** | 1024 | 1024 |
| **计算精度** | BF16 | BF16 |
| **有效单批次大小 (Batch Size)** | 4 (1$\times$4) | 8 (1$\times$8) |
| **学习率调度器** | Cosine | Cosine |
| **LoRA 目标模块 (Targets)** | q, k, v, o, gate, up, down | q, k, v, o, gate, up, down |
| **Dropout 概率** | 0.1 | 0.05 |
| **DPO 散度系数 (Beta)** | — | 0.05 |

---

## 评测结果

本章节展示了原始基座模型与微调阶段（SFT & DPO）在面对高难度硬性约束提示词时，最真实的终端日志（Terminal Logs）对比画面。

### 1. 原始基座模型全线失败（对照组）
在进行对齐训练之前，原生基座模型在单次连续运行中，连续无法满足格式约束、字符集约束以及词汇级边界约束：

<img src="images/base_all_fails.png" width="100%">

---

### 2. 递进式对齐演进效果（SFT vs. DPO）

#### 任务一：格式约束测试 — 电报文体风格
* **提示词 (Prompt):** *How do I make sure my Wi-Fi is secure? Construct the reply as if it's a telegram STOP.*

| 🟢 SFT 监督微调阶段（初见成效） | 👑 DPO 偏好对齐阶段（终极收敛） |
| :---: | :---: |
| **`sft_base`** 模型已成功捕获目标格式约束，输出规整的全大写文本并附带明确的 STOP 标记。 | **`dpo_v2_2`** 引入 LoRA 偏好对进一步约束概率分布，输出更稳定、严密的对齐结果。 |
| <img src="images/sft_telegram_pass.png" width="100%"> | <img src="images/dpo_telegram_pass.png" width="100%"> |

#### 任务二：词汇约束测试 — 必须使用以 "-ing" 结尾的单词
* **提示词 (Prompt):** *How to start a book club? Use words that end with '-ing'.*

| 👑 DPO 偏好对齐阶段（词尾约束完美通关） |
| :---: |
| 相比于基座模型发生幻觉输出非英语字词，最终的 DPO 模型在终端打印出的每个独立 Token 均 100% 满足 `-ing` 词尾约束。 |
| <img src="images/dpo_ing_pass.png" width="80%"> |

---

## 流水线统计数据

以下数据基于在 AutoDL 云服务器（单卡 NVIDIA A800 80 GB）上针对通用领域运行标准测试所得。

> **可复现性说明：** 由于大语言模型在文本生成过程中存在固有的随机性（Stochasticity），且 DPO 偏好对构建阶段限制了每个 Prompt 最多随机采样 2 对正负样本，实际运行时的各阶段绝对样本计数可能会有轻微上下浮动。

### 端到端完整数据流向

| 阶段 | 统计指标 | 样本数量 | 备注说明 |
| --- | --- | --- | --- |
| **Step 0** | 原始种子指令数 | 35 | 流水线的初始冷启动输入 |
| **Step 1** | RFT 扩充后的指令数 | 153 | 大幅泛化扩充后的主指令池 |
| **Step 2** | 进入代码验证阶段的指令 | 189 | 35 + 153 构成的完整评估集 |
| **Step 3** | 交叉验证留存的高可靠指令 | 70 | 剔除了无法通过代码自动化校验的指令 |
| **Step 5** | NLI 一致性过滤留存数 | 59 | 过滤后的最终留存率为 86.76% |
| **Step 6** | 多候选响应合成样本池 | 4,720 | 16倍 Query 扩充 $\times$ 每条生成 5 个 Candidate |
| **Step 7** | 通过 Python 执行验证的有效数 | 1,578 | 成功跑通验证逻辑的优质数据 |
| **Step 8** | 终审高质量 SFT 数据样本 | 27 | 严格筛选评分 Score > 8（满分 10 分）的样本 |
| **DPO** | 最终合成的 DPO 偏好对 | 587 | 基于 Step 6/7 全量池交叉配对生成的标准对齐数据 |

### 代码验证器拦截率统计 (Step 7)

在 Step 6 生成的 4,720 个候选大模型响应中：

* **通过 Python 真实代码执行验证的样本：** 1,578 个
* **被代码验证器拦截踢除的失败样本：** 3,142 个
* **验证器整体拦截率 (Filter Rejection Rate)：** **66.57%**

---

## 项目结构

```text
AutoIF-LLM/
├── README.md                           # 英文主文档
├── README_zh.md                        # 本文档（中文版）
├── setup.sh                            # 一键环境配置脚本
├── run_all.sh                          # 自动化主流水线运行脚本 (支持 --domain)
├── requirements.txt                    # Python 依赖依赖项列表
│
├── code_sft/                           # AutoIF 9步数据合成模块
│   ├── 1_RFT.py                        # Step 1: 指令扩充
│   ├── 2_verification_*.py             # Step 2: 自动化验证函数生成
│   ├── 3_cross_validation.py           # Step 3: 交叉验证过滤
│   ├── 4_eval_func_*.py                # Step 4: 语义反向翻译
│   ├── 5_eval_func_*_filter.py         # Step 5: NLI 一致性过滤
│   ├── 6_concat_sharegpt_*.py          # Step 6: Query 扩充与多候选响应生成
│   ├── 7_query_verification.py         # Step 7: 基于执行反馈的动态评分
│   ├── 8_query_score_filter.py         # Step 8: 高质量微调过滤
│   └── 9_sft_data_*.py                 # Step 9: SFT 数据集最终封装
│
├── code_dpo/                           # DPO 偏好数据构建模块
│   ├── 1_dpo_rft_wash.py               # 响应数据清洗与分档
│   └── 2_dpo_data_*.py                 # 偏好对自动交叉构建
│
├── scripts/                            # 辅助工具脚本目录
│   ├── generate_seed_instructions.py   # 领域专属种子指令生成器
│   ├── extended_domains.py             # 30+ 内置专业领域模板定义定义
│   ├── download_models.sh              # 游模型自动下载辅助脚本
│   └── patches/                        # 核心生态兼容性工程补丁包
│       ├── fix_qwen.py                 # 修复 vLLM 启动时的 rope_scaling 校验冲突
│       ├── fix_config.py               # 规范化并清洗各模型的非标准 RoPE 配置字段
│       ├── dpo_modification.py         # 动态向 LLaMA-Factory 注册 DPO 排序数据集
│       └── dpo2_modification.py        # 展平嵌套 ShareGPT 格式以适配原生 Alpaca 结构
│
├── configs/                            # LLaMA-Factory 训练超参数配置文件目录
│   ├── llamafactory_sft_lora.yaml      # SFT 阶段训练配置文件
│   ├── llamafactory_dpo_lora.yaml      # DPO 阶段训练配置文件
│   ├── llama_factory_dataset_info.json # 大模型训练数据集注册注册表
│   └── pipeline_config.yaml           # 全局流水线控制参数
│
├── sample_data/
│   └── seed_instruction.txt            # 系统默认内置的种子指令集（共 36 条）
│
├── models/                             # 上游游开源大模型权重目录（由 setup.sh 自动填充）
│
├── output/                             # 运行中间产物与最终模型导出目录
│   ├── IF_sft_data.json                # 精选的高质量第一阶段 SFT 数据集
│   └── dpo_pairs_flat.jsonl            # 标准 Alpaca 格式的第二阶段 DPO 偏好对
│
└── logs/                               # 各阶段系统执行 Logs 日志目录

```

---

## 故障排除与补丁

由于大模型开源生态（vLLM、LLaMA-Factory 及 Transformers）版本更迭异常频繁，在特定硬件环境下极易发生库函数冲突。AutoIF 在 `scripts/patches/` 目录下内置了 4 组工业级工程补丁，主脚本 `run_all.sh` 运行期间会自动调用并注入，如需手动逐步 Debug，可参照下表手动调用：

| 补丁脚本名称 | 触发报错条件 | 补丁解决逻辑 | 手动修复命令 |
| --- | --- | --- | --- |
| **fix_qwen.py** | 启动 vLLM 教师模型服务器失败，抛出 `rope_scaling` 强校验错误 | 动态在指定模型的 `config.json` 中注入通用的 `{"factor": 1.0, "type": "default"}` 规避检查 | `python scripts/patches/fix_qwen.py ./models/teacher` |
| **fix_config.py** | 训练框架读取上游模型配置失败，无法解析非标准位置编码 | 遍历并清洗 `models/` 目录下所有模型的非标准额外字段，确保训练框架正常读取 | `python scripts/patches/fix_config.py` |
| **dpo_modification.py** | LLaMA-Factory 训练器误将 DPO 偏好对识别为普通 SFT 数据 | 动态向 `dataset_info.json` 注入 `"ranking": true` 标签和正确的 ShareGPT 字段映射 | `python scripts/patches/dpo_modification.py` |
| **dpo2_modification.py** | 框架在解析复杂的嵌套多轮对话数组时发生句法错误 | 将多轮数据一键展平（Flatten）为更稳定的标准 Alpaca 格式，并重新在后端注册 | `python scripts/patches/dpo2_modification.py` |

---

## 参与贡献

我们非常欢迎学术界与开源社区同仁为本项目贡献代码。如有修改建议，请遵循标准的 GitHub 开发工作流：

1. Fork 本仓库。
2. 创建您的功能开发分支：`git checkout -b feature/your-feature-name`
3. 提交您的修改，请务必附带清晰直观的 Commit Messages。
4. 提交 Pull Request 至本仓库的 `main` 主分支。

若希望贡献新的垂直领域模板，请直接修改 `scripts/extended_domains.py` 并确保为该新领域提供至少 10 条具备代表性的初始种子指令。

---

## 论文引用

如果您在学术研究中使用了 AutoIF 框架，或者基于本项目开展了后续工作，请引用原始论文：

```bibtex
@article{dong2024self,
  title={Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models},
  author={Dong, Guanting and Lu, Keming and Li, Chengpeng and Xia, Tingyu and Yu, Bowen and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2406.13542},
  year={2024}
}

```

---

## 开源协议

本项目采用 **Apache License 2.0** 开源协议。

项目内涉及的各上游基础大模型（Qwen2.5 系列、mDeBERTa-v3）分别受其对应上游官方开源协议约束，商用前请严格审查上游协议条款。

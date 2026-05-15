# AutoIF: 基于自动指令遵循的大模型微调系统

> 基于论文 [Self-play with Execution Feedback](https://arxiv.org/abs/2406.13542) (ICLR 2025 Spotlight) 实现
> 支持**任意领域**（法律/金融/医疗/教育...）的大模型微调

---

## 项目亮点

- **领域无关**: 只需替换种子指令即可适配任何专业领域（法律、金融、医疗等）
- **全自动Pipeline**: 从数据生成到训练一键完成，无需人工标注
- **单卡可跑**: A800 单卡 80GB 即可运行完整流程
- **20分钟出结果**: 不含环境安装，全流程约20分钟

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA A800 80GB |
| 系统 | Ubuntu + Python 3.10+ |
| CUDA | 12.x |
| 磁盘 | 至少40GB可用空间 |

### Step 1: 上传代码

```bash
cd /root/autodl-tmp
# 如果您是通过压缩包上传的：
unzip AutoIF-LLM.zip && cd AutoIF-LLM

```

### Step 2: 一键安装环境

```bash
bash setup.sh

```

安装内容：

* PyTorch 2.4.0 + CUDA 12.1
* vLLM 0.5.5（推理加速）
* LlamaFactory（训练框架）
* 教师模型 Qwen2.5-7B-Instruct（15GB）
* 学生模型 Qwen2.5-1.5B（3GB）
* NLI模型 mDeBERTa-v3（2.5GB）

### Step 3: 一键运行全流程

```bash
# 通用领域
bash run_all.sh

# 指定领域
bash run_all.sh --domain 法律
bash run_all.sh --domain 金融
bash run_all.sh --domain 医疗

# 后台运行
nohup bash run_all.sh --domain 法律 > run.log 2>&1 &
tail -f run.log 

```

---

## 领域适配（核心创新点）

本项目的核心创新在于：**通过替换种子指令，自动生成任意领域的高质量训练数据**。

### 工作原理

```
种子指令 → AutoIF Pipeline → SFT数据 + DPO数据 → LoRA微调 → 领域大模型
   ↑                                                           ↓
 可替换！                                                  任意领域！

```

### 内置领域（30+）

| 分类 | 领域 |
| --- | --- |
| 基础科学 | 数学、物理、化学、生物、天文、地理 |
| 传统工程 | 土木工程、机械工程、电子工程、化工、材料科学、能源工程 |
| 人文社科 | 文学、历史、哲学、新闻传播、社会学、心理学 |
| 经济管理 | 工商管理、会计、公共管理、电子商务、金融 |
| 应用领域 | 法律、医疗、教育、编程 |
| 艺术体育 | 美术、音乐、体育 |

### 自定义领域

```bash
# 查看所有可用领域
python scripts/generate_seed_instructions.py --list

# 生成自定义领域的种子指令
python scripts/generate_seed_instructions.py --domain 建筑设计 --use-llm

# 使用自定义领域运行
bash run_all.sh --domain 建筑设计

```

### 领域适配示例

**法律大模型微调**：

```bash
bash run_all.sh --domain 法律
# 种子指令示例：
# - "引用具体的法律条文支持你的观点"
# - "区分民事责任和刑事责任"
# - "按时间顺序说明法律程序步骤"

```

**金融大模型微调**：

```bash
bash run_all.sh --domain 金融
# 种子指令示例：
# - "分析该投资产品的风险等级"
# - "计算该理财产品的预期收益率"
# - "对比不同投资策略的适用人群"

```

**医学大模型微调**：

```bash
bash run_all.sh --domain 医疗
# 种子指令示例：
# - "列举该疾病的常见症状（不少于5个）"
# - "对比不同治疗方案的优缺点"
# - "说明该药物的用法用量和注意事项"

```

---

## 技术架构

### 全流程概览

```
阶段1: vLLM教师模型启动
  └── Qwen2.5-7B-Instruct 作为数据生成的教师

阶段2: AutoIF 数据合成（9步）
  ├── Step 1: 指令增强（36→189条）
  ├── Step 2: 验证函数生成
  ├── Step 3: 交叉验证（质量过滤）
  ├── Step 4: 反向翻译
  ├── Step 5: NLI 一致性过滤
  ├── Step 6: 查询增强 + 响应生成
  ├── Step 7: 质量评分
  ├── Step 8: 质量过滤（Score > 8）
  └── Step 9: SFT 数据构建

阶段3: DPO 数据构建
  ├── DPO-1: 响应评分
  └── DPO-2: 偏好对构建

阶段4: SFT 训练（LoRA）
阶段5: DPO 训练（LoRA）
阶段6: LoRA 权重合并
阶段7: 量化
阶段8: 推理测试

```

### 模型配置

| 角色 | 模型 | 大小 | 用途 |
| --- | --- | --- | --- |
| 教师模型 | Qwen2.5-7B-Instruct | 15GB | 生成训练数据 |
| 学生模型 | Qwen2.5-1.5B | 3GB | 训练目标模型 |
| NLI 模型 | mDeBERTa-v3 | 2.5GB | 指令一致性过滤 |

### 训练参数

| 参数 | SFT 阶段 | DPO 阶段 |
| --- | --- | --- |
| **微调方法** | LoRA (rank=32, alpha=64) | LoRA (rank=8, alpha=16) |
| **学习率** | 1e-4 | 3e-6 |
| **训练轮数 (Epochs)** | 15 | 5 |
| **最大长度 (Cutoff Len)** | 1024 | 1024 |
| **计算精度** | bf16 | bf16 |
| **Batch Size (等效)** | 4 (1x4) | 8 (1x8) |
| **调度器 (Scheduler)** | cosine | cosine |
| **LoRA 目标模块** | 全量 Projection (q,v,k,o,gate,up,down) | 全量 Projection (q,v,k,o,gate,up,down) |
| **Dropout** | 0.1 | 0.05 |
| **Beta (DPO 专属)** | - | 0.05 |

---

## 📊 实验效果对比 (Model Evaluation)

本项目对微调前后的模型进行了严格的 **指令遵循约束** 测试。以下是 Base 模型与微调对齐后模型在面对复杂格式约束、负向约束时的真实表现对比：
### 1. 格式约束：电报风格 (Telegram Style)
* **测试指令**：`How do I make sure my Wi-Fi is secure? Construct the reply as if it's a telegram STOP.`
* **对齐表现**：

| 模型阶段 | 真实输出 (Actual Output) | 结果评估 |
|:---|:---|:---|
| **Base 模型** | `Here's how you can make your Wi-Fi network more secure:\n1. Change your Wi-Fi password...` | ❌ **失败**：完全忽略了"电报风格"和"STOP"约束，输出了常规的1-8点普通列表。 |
| **SFT/DPO 模型** | `BEGIN TELEGRAM STOP TO ENSURE YOUR WIFI SECURITY IMPLEMENT STRONG PASSWORDS... STOP END TELEGRAM STOP` | ✅ **成功**：完美遵循全大写电报格式，并在内部断句及末尾精准嵌入了 `STOP` 符号。 |

### 2. 词汇负向约束：必须以 '-ing' 结尾
* **测试指令**：`How to start a book club? Use words that end with '-ing'.`
* **对齐表现**：

| 模型阶段 | 真实输出 (Actual Output) | 结果评估 |
|:---|:---|:---|
| **Base 模型** | `woordend met de klasse '-ing'` | ❌ **失败**：产生严重的语种幻觉，输出了非英文的无意义拼凑字符。 |
| **DPO 模型** | `Creating, inviting, discussing, reading, scheduling, meeting, planning, reviewing...` | ✅ **成功**：不仅回答内容完全切题，且输出的每一个单词都严格满足 `-ing` 结尾的硬性代码逻辑限制。 |

### 3. 高级字符集约束：仅使用前半部分字母 (A-M)

* **测试指令**：`Explain NLP briefly. Use only the first half of the alphabet (A-M).`
* **Base 模型表现**：直接输出了包含大量 `N~Z` 范围外部字母的长篇大论，彻底暴露出原生开源大模型无法理解底层 Token/字符级负向约束的固有缺陷。

---

## 📈 数据流水线审计与统计 (Pipeline Statistics)

以下基于 **AutoDL (NVIDIA A800 80GB)** 平台运行通用领域微调的完整数据流转审计。

> ⚠️ **复现说明（关于数据随机性）**：
> 由于大语言模型（LLM）在生成过程中存在固有的**随机性（Stochasticity）**，且 DPO 阶段包含随机采样（$\le 2$ 个正/负样本组合），**不同批次运行产生的绝对数量和留存率会有轻微浮动**。以下数据为典型基准（Benchmark）运行报告。

### 1. 全流程数据流转与留存追踪

| 阶段 (Stage) | 核心节点与指标 | 实例数量 (Count) | 流水线状态 / 留存与修正说明 |
| --- | --- | --- | --- |
| **Step 0** | 原始种子指令数 (Seed) | 35 条 | 初始种子输入 |
| **Step 1** | RFT 指令增强 | 153 条 | 扩充指令池拓扑 |
| **Step 2** | 用于验证的指令总数 | 189 条 | $35 + 153$ 全量进入物理代码验证阶段 |
| **Step 3** | 交叉验证幸存指令 | 70 条 | 仅保留在物理测试环境中具备高可靠判别器的指令 |
| **Step 5** | NLI 一致性过滤留存 | 59 条 | **语义一致性留存率：86.76%**（淘汰9条矛盾指令） |
| **Step 6** | 数据裂变（查询与响应） | 4,720 条 | 幸存指令进行 16× 扩流与 5× 响应生成 |
| **Step 7** | 代码格式与执行反馈审计 | 1,578 条 | 经历物理执行测试后，未发生严重格式崩溃的有效样本 |
| **Step 8** | SFT 高质量筛选 | **27 条** | 经过教师模型多维度评分，筛选出分值 >8 分（即10分）的极端极品样本，构建 `IF_sft_data.json`。 |
| **DPO 阶段** | 偏好对构建 (DPO-1/2) | **587 对** | **【架构修正】** 数据并非来自 Step 8，而是回溯至 Step 6/7 的 4720 条全量响应，利用代码验证通过率构建 `dpo_pairs_flat.jsonl`。 |

### 2. 核心清洗阶段深度审计

#### 💥 Python 执行反馈拦截率（Step 7）

模型生成的 4,720 条候选响应面临了真实的 Python 环境自动化测试拦截：

* 通过 Python 格式与逻辑审计的样本数：1,578
* 被 Python 函数拒绝的低质样本数：3,142
* **格式过滤杀伤率：66.57%**

#### 💡 DPO 核心数据流洞察

在构建 `587 对` DPO 偏好数据时，系统采取了**回溯对立策略**：

1. **数据源回溯**：DPO 偏好对的构建直接绕过了 Step 8 的高分过滤，重新激活了 Step 6/7 生成的全部响应。
2. **正负样本分离**：利用代码判别器的物理执行通过率（Accuracy Score）作为硬性指标：
* 🌟 **正样本 (Chosen)**：通过率 $\ge 0.7$ 的响应。
* 🚨 **负样本 (Rejected)**：通过率 $= 0$ 的响应。


3. **组合配对条件**：同一个 prompt 下必须同时存在至少 1 个正样本和 1 个负样本。从正负池中各随机采样 $\le 2$ 个建立笛卡尔积对立对。
4. **核心科学依据**：DPO（直接偏好优化）的本质是学习“好坏对比”。在 SFT 阶段被无情淘汰的 3,142 条“垃圾数据”，在这里恰恰是绝佳的负面教材。这种设计成功将负产物转化为对齐资产，拉大了偏好 Margin 差值。

---

## 📦 高质量精选数据集样例 (Curated Dataset Showcase)

本项目之所以能取得显著的强约束对齐效果，核心得益于流水线生成的精选数据集。以下为保存在 `output/IF_sft_data.json` 中的代表性核心训练样本：

```json
[
  {
    "instruction": "What are the key concepts in natural language processing? Construct the reply as if it's a telegram STOP.",
    "output": "KEY CONCEPTS IN NATURAL LANGUAGE PROCESSING INCLUDE MACHINE LEARNING ALGORITHMS, TOKENIZATION... STOP",
    "history": []
  },
  {
    "instruction": "How to start a book club? Use words that end with '-ing'.",
    "output": "Creating, inviting, discussing, reading, scheduling, meeting, planning, reviewing, enjoying, sharing, analyzing, engaging.",
    "history": []
  },
  {
    "instruction": "How to troubleshoot slow internet? Write the response with words in alphabetical order.",
    "output": "check, connect, disable, download, ensure, fix, identify, isolate, reboot, speed, test, upgrade",
    "history": []
  }
]

```

> 💡 **数据集亮点**：合成数据包含了电报风格、指定词尾（-ing）、首字母范围限制（A-M）、纯疑问句交互、乃至**极端难度的全文本单词按字母表顺序升序排列（Alphabetical Order）**。这类高质量指令样本在当前的公用数据集中极度稀缺，是本项目实现领域大模型精准长尾对齐的基石。

---

## 项目结构

```
├── README.md                    # 本文件
├── setup.sh                     # 一键环境安装
├── run_all.sh                   # 一键运行全流程（支持 --domain）
├── requirements.txt             # Python 依赖
├── code_sft/                        # AutoIF 数据合成代码（9步）
│   ├── 1_RFT.py                 # Step 1: 指令增强
│   ├── 2_verification_*.py      # Step 2: 验证函数生成
│   ├── 3_cross_validation.py    # Step 3: 交叉验证
│   ├── 4_eval_func_*.py         # Step 4: 反向翻译
│   ├── 5_eval_func_*_filter.py  # Step 5: NLI一致性过滤
│   ├── 6_concat_sharegpt_*.py   # Step 6: 查询+响应生成
│   ├── 7_query_verification.py  # Step 7: 质量评分
│   ├── 8_query_score_filter.py  # Step 8: 质量过滤
│   ├── 9_sft_data_*.py          # Step 9: SFT数据构建
│   └── utils.py                 # 工具函数（LLM调用等）
├── code_dpo/                    # DPO 数据构建代码
│   ├── 1_dpo_rft_wash.py        # 响应评分
│   └── 2_dpo_data_*.py          # 偏好对构建
├── scripts/                     # 辅助脚本
│   ├── generate_seed_instructions.py  # 领域种子指令生成器
│   ├── extended_domains.py      # 30+领域模板定义
│   └── download_models.sh       # 模型下载脚本
├── configs/                     # LlamaFactory训练配置
│   ├── llamafactory_sft_lora.yaml   # SFT训练配置
│   ├── llamafactory_dpo_lora.yaml   # DPO训练配置
│   ├── llama_factory_dataset_info.json
│   └── pipeline_config.yaml     # 流水线参数配置
├── sample_data/                 # 种子数据
│   └── seed_instruction.txt     # 默认种子指令（36条）
├── models/                      # 模型目录（setup.sh自动下载）
├── output/                      # 运行时输出数据
│   ├── IF_sft_data.json         # 精选SFT高质量指令数据集
│   └── dpo_pairs_flat.jsonl     # 精选DPO偏好对数据集（Alpaca格式）
├── logs/                        # 运行日志

```

---

## 🔧 兼容性补丁与故障排除 (On-demand Patches)

由于大模型生态中 `vLLM`、`LLaMA-Factory` 以及 `Transformers` 版本迭代迅速，硬件环境（如 AutoDL A800）可能会由于底层库版本冲突引发初始化错误。

本项目在 `scripts/patches/` 目录中内置了 4 个自动化修复补丁。**通常情况下，全流程一键运行脚本会自动处理，若您手动分步调试遭遇异常，可按需运行以下补丁：**

### 1. 模型配置与推理适配补丁

| 补丁脚本 | 针对问题 | 修复原理 | 运行命令 |
| --- | --- | --- | --- |
| `fix_qwen.py` | 启动 vLLM 教师模型时报错，提示 `rope_scaling` 配置缺失或校验失败。 | 动态在指定模型的 `config.json` 中强行注入 vLLM 所需的 `{"factor": 1.0, "type": "default"}` 算子参数。 | `python scripts/patches/fix_qwen.py ./models/teacher` |
| `fix_config.py` | 启动 SFT/DPO 训练时，框架因无法解析模型中的未知位置编码缩放而引发崩溃。 | 全局遍历 `models/` 目录，安全剥离所有非必要的 `rope_scaling` 字段，确保训练框架正常初始化。 | `python scripts/patches/fix_config.py` |

### 2. 训练框架数据集注册补丁

| 补丁脚本 | 针对问题 | 修复原理 | 运行命令 |
| --- | --- | --- | --- |
| `dpo_modification.py` | LLaMA-Factory 无法正确识别 DPO 偏好对数据，将其误认作标准 SFT 数据。 | 在 LLaMA-Factory 的 `dataset_info.json` 中强制注入 `"ranking": true` 标志，并建立 **ShareGPT** 格式映射。 | `python scripts/patches/dpo_modification.py` |
| `dpo2_modification.py` | 训练框架在解析复杂的 ShareGPT 对话嵌套格式时产生解析异常或效率低下。 | **【格式升级】** 提取原始数据中的 `value` 文本，将 DPO 数据全量**扁平化（Flatten）**，重构并注册为更稳定的 **Alpaca** 外层格式。 | `python scripts/patches/dpo2_modification.py` |

---

## 参考文献

```bibtex
@article{dong2024self,
  title={Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models},
  author={Dong, Guanting and Lu, Keming and Li, Chengpeng and Xia, Tingyu and Yu, Bowen and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2406.13542},
  year={2024}
}

```
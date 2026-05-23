#!/bin/bash
# AutoIF 项目一键环境配置脚本（AutoDL A800）

set -e

export PATH=/root/miniconda3/bin:$PATH

echo "=========================================="
echo "  AutoIF 项目环境配置"
echo "=========================================="

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo "[1/4] 安装训练环境依赖（base）..."
pip install -r requirements_train.txt $PIP_MIRROR -q
echo "✅ 训练依赖安装完成"

echo "[2/4] 安装 LlamaFactory..."
if [ ! -d "LlamaFactory" ]; then
    git clone --depth 1 https://gitee.com/hiyouga/LLaMA-Factory.git LlamaFactory 2>/dev/null || \
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LlamaFactory 2>/dev/null || true
fi
cd LlamaFactory && pip install -e ".[torch,metrics]" -q $PIP_MIRROR 2>/dev/null && cd ..
echo "✅ LlamaFactory 安装完成"

echo "[3/4] 注册数据集到 LlamaFactory..."
if [ -d "LlamaFactory/data" ] && [ -f "configs/llama_factory_dataset_info.json" ]; then
    python -c "
import json
lf_path = 'LlamaFactory/data/dataset_info.json'
with open(lf_path, 'r', encoding='utf-8') as f:
    lf_data = json.load(f)
with open('configs/llama_factory_dataset_info.json', 'r', encoding='utf-8') as f:
    custom_data = json.load(f)
lf_data.update(custom_data)
with open(lf_path, 'w', encoding='utf-8') as f:
    json.dump(lf_data, f, indent=2, ensure_ascii=False)
"
    echo "✅ 数据集注册完成"
fi

echo "[4/4] 创建必要目录..."
mkdir -p models/student output logs
echo "✅ 训练环境配置完成！使用 scripts/download_models.sh 补充模型，然后运行 scripts/run_all.sh"
echo ""
echo "  📦 附加环境说明（按需创建）："
echo ""
echo "  ① gptq_env — 用于 vLLM 高并发推理（推荐，同时支持 Base/SFT/DPO/GPTQ 模型）："
echo "    conda create -n gptq_env python=3.10 -y"
echo "    conda activate gptq_env"
echo "    pip install -r requirements_gptq_vllm.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
echo "    # 之后运行: python tests/evaluate_vllm.py"
echo ""
echo "  ② hf_eval — 用于在原生 Transformers 下单独评测 GPTQ 模型（与 base 环境有依赖冲突）："
echo "    conda create -n hf_eval python=3.10 -y"
echo "    conda activate hf_eval"
echo "    pip install -r requirments_GPTQ_model_hf_eval.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
echo "    # 运行前请将 tests/evaluate_hf_batched.py 中 models_to_test 改为只保留 GPTQ-Model"
echo "    # 之后运行: python tests/evaluate_hf_batched.py"
echo ""
echo "  ⚠️  注意：hf_eval 与 base 环境存在依赖冲突，请勿混用。"
echo "      GPTQ 在原生 Transformers 下速度较慢（约 45 tokens/s），"
echo "      推荐优先使用 gptq_env + vLLM 方案（约 1482 tokens/s）。"
echo "=========================================="
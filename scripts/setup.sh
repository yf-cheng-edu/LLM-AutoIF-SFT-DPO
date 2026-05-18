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

echo "[1/4] 安装 Python 依赖..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 $PIP_MIRROR -q
pip install -q $PIP_MIRROR transformers accelerate peft trl datasets openai jsonlines tqdm tenacity modelscope huggingface-hub safetensors tiktoken
pip install vllm==0.5.5 -q $PIP_MIRROR
echo "✅ 依赖安装完成"

echo "[2/4] 安装 LlamaFactory..."
if [ ! -d "LlamaFactory" ]; then
    git clone --depth 1 https://gitee.com/hiyouga/LLaMA-Factory.git LlamaFactory 2>/dev/null || \
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LlamaFactory 2>/dev/null || true
fi
cd LlamaFactory && pip install -e ".[torch,metrics]" -q $PIP_MIRROR 2>/dev/null && cd ..
echo "✅ LlamaFactory 安装完成"

echo "[3/4] 注册数据集到 LlamaFactory..."
# 将你写好的配置合并进去
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
echo "✅ 环境配置完成！使用 scripts/download_models.sh 补充模型，然后运行 run_all.sh"
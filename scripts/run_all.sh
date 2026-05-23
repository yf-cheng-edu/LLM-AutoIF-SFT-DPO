#!/bin/bash
# AutoIF 项目一键运行全流程
# 前置条件: 已运行 setup.sh
# 用法:
#   bash scripts/run_all.sh                    # 通用领域
#   bash scripts/run_all.sh --domain 法律       # 法律领域微调
#   bash scripts/run_all.sh --domain 金融       # 金融领域微调

set -e

export PATH=/root/miniconda3/bin:$PATH

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

# ========== 解析命令行参数 ==========
DOMAIN="通用"
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================"
echo "  AutoIF 全流程一键运行"
echo "  领域: $DOMAIN"
echo "  开始时间: $(date)"
echo "============================================"

# ========== 领域适配: 生成或切换种子指令 ==========
if [ "$DOMAIN" != "通用" ]; then
    SEED_FILE="sample_data/seed_instruction_${DOMAIN}.txt"
    if [ ! -f "$SEED_FILE" ]; then
        echo "生成 $DOMAIN 领域种子指令..."
        python scripts/generate_seed_instructions.py --domain "$DOMAIN" --count 30 --output "$SEED_FILE"
    fi
    # 备份原始种子指令，替换为领域指令
    cp sample_data/seed_instruction.txt sample_data/seed_instruction_backup.txt 2>/dev/null || true
    cp "$SEED_FILE" sample_data/seed_instruction.txt
    echo "✅ 已切换到 $DOMAIN 领域种子指令（$(wc -l < sample_data/seed_instruction.txt) 条）"
fi

# ========== 环境变量配置 (指向 DeepSeek API) ==========
export SUPERVISOR_API_BASE="https://api.deepseek.com/v1"
export SUPERVISOR_API_KEY="填写你的真实API_KEY" # 注意：请确保此处或环境中已有你的真实 Key
export SUPERVISOR_MODEL="deepseek-chat"
export NLI_MODEL_PATH="./models/nli"
export HF_ENDPOINT="https://hf-mirror.com"

STUDENT_PATH="$PROJECT_DIR/models/student/Qwen/Qwen2.5-1.5B-Instruct"

mkdir -p output logs

# ================================================================
# 阶段 1: AutoIF 数据合成 
# ================================================================
echo ""
echo "============================================"
echo "  阶段 1: AutoIF 数据合成 (调用 DeepSeek)"
echo "============================================"

echo "[Step 1/9] 指令增强..."
python code_sft/1_RFT.py 2>&1 | tee logs/step1.log

echo "[Step 2/9] 验证函数生成..."
python code_sft/2_verification_funcs_cases_generation.py 2>&1 | tee logs/step2.log

echo "[Step 3/9] 交叉验证..."
python code_sft/3_cross_validation.py 2>&1 | tee logs/step3.log

echo "[Step 4/9] 反向翻译..."
python code_sft/4_eval_func_backtranslator.py 2>&1 | tee logs/step4.log

echo "[Step 5/9] NLI 一致性过滤..."
python code_sft/5_eval_func_backtranslator_filter.py 2>&1 | tee logs/step5.log

echo "[Step 6/9] 查询增强与响应生成..."
python code_sft/6_concat_sharegpt_query.py 2>&1 | tee logs/step6.log
python tools/extract_test_set.py 2>&1 | tee logs/step6.log
echo "[Step 7/9] 查询验证与质量评分..."
python code_sft/7_query_verification.py 2>&1 | tee logs/step7.log

echo "[Step 8/9] 质量过滤..."
python code_sft/8_query_score_filter.py 2>&1 | tee logs/step8.log

echo "[Step 9/9] SFT 数据构建..."
python code_sft/9_sft_data_construction.py 2>&1 | tee logs/step9.log

echo "✅ SFT 数据合成完成"

echo ""
echo "[DPO-1] 响应评分..."
python code_dpo/1_dpo_rft_wash.py 2>&1 | tee logs/dpo1.log

echo "[DPO-2] 偏好对构建..."
python code_dpo/2_dpo_data_query_construct.py 2>&1 | tee logs/dpo2.log

# DPO 数据展平修复 (替代 dpo2_patches.py)
echo "[DPO-3] 展平 DPO 数据格式..."
python -c "
import json
input_file = 'output/dpo_pairs.jsonl'
output_file = 'output/dpo_pairs_flat.jsonl'
with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
    for line in f:
        data = json.loads(line)
        prompt_text = data['conversations'][0]['value']
        new_item = {'prompt': prompt_text, 'chosen': data['chosen'], 'rejected': data['rejected']}
        out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
"
echo "✅ DPO 数据合成与格式修正完成"


# ================================================================
# 阶段 2: SFT 训练与合并
# ================================================================
echo ""
echo "============================================"
echo "  阶段 2: SFT 训练"
echo "============================================"
cd LlamaFactory
cd LlamaFactory
llamafactory-cli train \
  --model_name_or_path "$STUDENT_PATH" \
  --stage sft \
  --do_train \
  --dataset autoif_sft \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir "$PROJECT_DIR/models/model_d_sft" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --cutoff_len 2048 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --bf16 \
  --save_steps 150 \
  --logging_steps 5 \
  --plot_loss \
  --val_size 0.1 \
  --eval_strategy steps \
  --eval_steps 50 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir 2>&1 | tee "$PROJECT_DIR/logs/sft_train.log"
  
echo "============================================"
echo "  阶段 3: SFT 模型合并"
echo "============================================"
llamafactory-cli export \
  --model_name_or_path "$STUDENT_PATH" \
  --adapter_name_or_path "$PROJECT_DIR/models/model_d_sft" \
  --export_dir "$PROJECT_DIR/models/model_d_sft_merged" \
  --finetuning_type lora \
  --template qwen 2>&1 | tee "$PROJECT_DIR/logs/sft_merge.log"

echo ""
echo "============================================"
echo "  阶段 4: DPO 训练"
echo "============================================"
llamafactory-cli train \
  --model_name_or_path "$PROJECT_DIR/models/model_d_sft_merged" \
  --stage dpo \
  --do_train \
  --dataset autoif_dpo \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir ../models/model_d_dpo_2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 2 \
  --cutoff_len 2048 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --save_steps 25 \
  --logging_steps 5 \
  --plot_loss \
  --pref_beta 0.3 \
  --val_size 0.1 \
  --eval_strategy steps \
  --eval_steps 25 \
  --per_device_eval_batch_size 2 \
  --overwrite_output_dir 2>&1 | tee "$PROJECT_DIR/logs/dpo_train.log"

echo "============================================"
echo "  阶段 5: DPO 模型合并 (使用最佳 Checkpoint-175)"
echo "============================================"
llamafactory-cli export \
    --model_name_or_path "$PROJECT_DIR/models/model_d_sft_merged" \
    --adapter_name_or_path "$PROJECT_DIR/models/model_d_dpo_2/checkpoint-175" \
    --export_dir "$PROJECT_DIR/models/model_d_dpo_merged" \
    --finetuning_type lora \
    --template qwen 2>&1 | tee "$PROJECT_DIR/logs/dpo_merge.log"


cd "$PROJECT_DIR"


echo ""
echo "============================================"
echo "  阶段 6: 基础/SFT/DPO 模型效果比对"
echo "============================================"
python patches/fix_config.py
python patches/fix_qwen.py "$PROJECT_DIR/models/model_d_dpo_merged"
python tests/models_to_test.py 2>&1 | tee logs/offline_test.log

echo ""
echo "============================================"
echo "  阶段 7: 虚拟环境配置与 GPTQ INT4 模型量化"
echo "============================================"

# 1. 创建并激活虚拟环境
eval "$(conda shell.bash hook)"
conda activate gptq_env

# 4. 源码编译 auto-gptq (强制本地编译以适配 A800 CUDA 环境)
BUILD_CUDA_EXT=1 pip install auto-gptq -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "  📦 正在对 DPO 合并模型进行 INT4 量化 (这可能需要几分钟)..."

python tests/GPTQ.py 2>&1 | tee "$PROJECT_DIR/logs/gptq_quant.log"
echo "  ✅ 量化完成！"

echo ""
echo "============================================"
echo "  阶段 8: vLLM INT4 服务部署与接口测试"
echo "============================================"
# 生成 Qwen 的 ChatML 对话模板
cat << 'EOF' > configs/chatml.jinja
{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}
EOF

# 启动 vLLM 在后台运行 
vllm serve models/model_d_dpo_merged_gptq_int4 \
    --quantization gptq \
    --dtype float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name qwen \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --chat-template configs/chatml.jinja > logs/vllm_serve.log 2>&1 &
VLLM_PID=$!

echo "  ⏳ 正在等待 vLLM 服务启动（预计 30-60 秒）..."
for i in {1..15}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "  ✅ vLLM 服务已就绪！"
        break
    fi
    sleep 5
done

# 运行接口测试脚本
python tests/test_vllm.py 2>&1 | tee logs/vllm_test.log

# 测试结束后关闭服务释放显存
echo "  🛑 测试完毕，正在关闭 vLLM 进程..."
kill $VLLM_PID || true


echo ""
echo "=========================================="
echo "  Step 10: Transformers 批处理评测（Base / SFT / DPO）"
echo "=========================================="
python tests/evaluate_hf_batched.py
echo "✅ HF 批处理评测完成 -> output/dpo_alignment_compare_hf_batched.json"

# ============================================================
# Step 11（可选）: GPTQ 模型评测（需单独的 hf_eval 环境）
# ============================================================
# GPTQ 与 base 环境存在依赖冲突，需在独立 conda 环境中运行。
# 请手动执行以下步骤：
#
#   conda create -n hf_eval python=3.10 -y
#   conda activate hf_eval
#   pip install -r requirments_GPTQ_model_hf_eval.txt \
#       -i https://pypi.tuna.tsinghua.edu.cn/simple
#
# 然后将 tests/evaluate_hf_batched.py 中 models_to_test 改为只保留
# GPTQ-Model 那一项，再运行：
#   python tests/evaluate_hf_batched.py
#
# 注意：GPTQ 在原生 Transformers 下速度较慢（约 45 tokens/s），
# 这是计算图额外开销 + 内存访存瓶颈导致的，属于正常现象。
# 推荐改用 Step 12 的 vLLM 方案获得最优速度（约 1482 tokens/s）。
echo ""
echo "⚠️  Step 11 (GPTQ/hf_eval): 需手动切换环境，已跳过。详见上方注释。"


# vLLM 同时支持 Base / SFT / DPO / GPTQ，且 GPTQ 速度提升约 33x。
# 需在 gptq_env 环境中运行（已由 setup.sh 提示创建）：
#
#   conda activate gptq_env
#   python tests/evaluate_vllm.py
#
echo "=========================================="
echo "  Step 12: vLLM 高并发评测（gptq_env）"
echo "=========================================="
if conda run -n gptq_env python -c "import vllm" 2>/dev/null; then
    conda run -n gptq_env python tests/evaluate_vllm.py
    echo "✅ vLLM 评测完成 -> output/vllm_dpo_alignment_compare.json"
else
    echo "⚠️  未检测到 gptq_env 环境或 vllm 未安装，跳过 Step 13。"
    echo "    请手动执行："
    echo "      conda activate gptq_env"
    echo "      python tests/evaluate_vllm.py"
fi

echo ""
echo "=========================================="
echo "  Step 14: LLM-as-a-Judge 全面评测"
echo "=========================================="
# 前置条件：llm_judge_all.py 中的 YOUR_API_KEY 已替换为真实 DeepSeek Key
python tools/llm_judge_all.py
echo "✅ 评测完成 -> output/all_models_judge_results.json"


echo ""
echo "============================================"
echo "  🎉 全流程运行完毕！领域: $DOMAIN"
echo "============================================"
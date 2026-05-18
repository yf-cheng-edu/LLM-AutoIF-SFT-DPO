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

STUDENT_PATH="../models/student/Qwen/Qwen2.5-1.5B-Instruct"

mkdir -p output logs

# ================================================================
# 阶段 1: AutoIF 数据合成 (通过 DeepSeek API)
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
  --output_dir ../models/model_d_sft \
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
  --overwrite_output_dir 2>&1 | tee ../logs/sft_train.log

echo "============================================"
echo "  阶段 3: SFT 模型合并"
echo "============================================"
llamafactory-cli export \
  --model_name_or_path "$STUDENT_PATH" \
  --adapter_name_or_path ../models/model_d_sft \
  --export_dir ../models/model_d_sft_merged \
  --finetuning_type lora \
  --template qwen 2>&1 | tee ../logs/sft_merge.log


# ================================================================
# 阶段 4: DPO 训练与合并
# ================================================================
echo ""
echo "============================================"
echo "  阶段 4: DPO 训练"
echo "============================================"
llamafactory-cli train \
  --model_name_or_path ../models/model_d_sft_merged \
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
  --overwrite_output_dir 2>&1 | tee ../logs/dpo_train.log

echo "============================================"
echo "  阶段 5: DPO 模型合并 (使用最佳 Checkpoint-175)"
echo "============================================"
llamafactory-cli export \
    --model_name_or_path ../models/model_d_sft_merged \
    --adapter_name_or_path ../models/model_d_dpo_2/checkpoint-175 \
    --export_dir ../models/model_d_dpo_merged \
    --finetuning_type lora \
    --template qwen 2>&1 | tee ../logs/dpo_merge.log

cd ..


# ================================================================
# 阶段 6: 测试与部署
# ================================================================
echo ""
echo "============================================"
echo "  阶段 6: 基础/SFT/DPO 模型离线效果比对"
echo "============================================"
python tests/models_to_test.py 2>&1 | tee logs/offline_test.log

echo ""
echo "============================================"
echo "  阶段 7: vLLM 服务部署与接口测试"
echo "============================================"
echo "  🩹 正在应用 Qwen 模型 rope_scaling 补丁以防 vLLM 崩溃..."
python patches/fix_config.py
python patches/fix_qwen.py models/model_d_dpo_merged

# 启动 vLLM 在后台运行
vllm serve models/model_d_dpo_merged \
    --dtype bfloat16 \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name qwen \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 > logs/vllm_serve.log 2>&1 &
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
echo "============================================"
echo "  🎉 全流程运行完毕！领域: $DOMAIN"
echo "============================================"
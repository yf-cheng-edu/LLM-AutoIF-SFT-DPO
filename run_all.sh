#!/bin/bash
# AutoIF 项目一键运行全流程
# 前置条件: 已运行 bash setup.sh
# 用法:
#   bash run_all.sh                    # 通用领域
#   bash run_all.sh --domain 法律       # 法律领域微调
#   bash run_all.sh --domain 金融       # 金融领域微调
#   bash run_all.sh --domain 医疗       # 医疗领域微调

set -e

export PATH=/root/miniconda3/bin:$PATH

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

# 解析命令行参数
DOMAIN="通用"
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo ""
echo "============================================"
echo "  AutoIF 全流程一键运行"
echo "  领域: $DOMAIN"
echo "  开始: $(date)"
echo "============================================"

# 解析模型实际路径（modelscope下载的目录结构）
TEACHER_PATH=$(find models/teacher -name "config.json" -path "*/Qwen*" | head -1 | xargs dirname 2>/dev/null || echo "models/teacher")
STUDENT_PATH=$(find models/student -name "config.json" -path "*/Qwen*" | head -1 | xargs dirname 2>/dev/null || echo "models/student")

echo "教师模型: $TEACHER_PATH"
echo "学生模型: $STUDENT_PATH"

# 环境变量
export SUPERVISOR_API_BASE="http://localhost:8000/v1"
export SUPERVISOR_API_KEY="EMPTY"
export SUPERVISOR_MODEL="Qwen/Qwen2.5-7B-Instruct"
export NLI_MODEL_PATH="./models/nli"
export HF_ENDPOINT="https://hf-mirror.com"

mkdir -p output logs

# 领域适配: 生成或切换种子指令
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

# ================================================================
# 阶段1: 启动 vLLM 教师模型
# ================================================================
echo ""
echo "============================================"
echo "  阶段1: 启动 vLLM 教师模型"
echo "============================================"

# 检查 vLLM 是否已在运行
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM 服务已在运行"
else
    echo "启动 vLLM 服务..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$TEACHER_PATH" \
        --served-model-name Qwen/Qwen2.5-7B-Instruct \
        --port 8000 \
        --trust-remote-code \
        --gpu-memory-utilization 0.5 \
        --max-model-len 4096 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # 等待 vLLM 启动
    echo "等待 vLLM 服务就绪..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ vLLM 服务就绪（等待 ${i}s）"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "❌ vLLM 启动超时"
            exit 1
        fi
        sleep 5
    done
fi

# ================================================================
# 阶段2: AutoIF 数据合成（9步）
# ================================================================
echo ""
echo "============================================"
echo "  阶段2: AutoIF 数据合成"
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

echo "✅ AutoIF 数据合成完成"

# DPO 数据构建
echo ""
echo "[DPO-1] 响应评分..."
python code_dpo/1_dpo_rft_wash.py 2>&1 | tee logs/dpo1.log

echo "[DPO-2] 偏好对构建..."
python code_dpo/2_dpo_data_query_construct.py 2>&1 | tee logs/dpo2.log

echo "✅ DPO 数据构建完成"

# 数据统计
echo ""
echo "--- 数据统计 ---"
for f in output/*.json output/*.jsonl; do
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        echo "  $(basename $f): $count 行"
    fi
done

# ================================================================
# 阶段3: 关闭 vLLM（释放显存给训练）
# ================================================================
echo ""
echo "关闭 vLLM 服务（释放显存）..."
if [ ! -z "$VLLM_PID" ]; then
    kill $VLLM_PID 2>/dev/null || true
fi
# 也杀掉其他可能的 vLLM 进程
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5
echo "✅ vLLM 已关闭"

# ================================================================
# 阶段4: SFT 训练
# ================================================================
echo ""
echo "============================================"
echo "  阶段4: SFT 训练（LoRA）"
echo "============================================"

# 准备 SFT 数据（转换为 LlamaFactory 格式）
python -c "
import json, os
sft_data = []
with open('output/query_score_filter.jsonl') as f:
    for line in f:
        item = json.loads(line)
        ans = item.get('gpt-answer', [''])
        output = ans[0] if isinstance(ans, list) and ans else str(ans)
        sft_data.append({'instruction': item.get('instruction',''), 'input': item.get('query',''), 'output': output})
print(f'SFT 数据: {len(sft_data)} 条')
os.makedirs('LlamaFactory/data', exist_ok=True)
with open('LlamaFactory/data/autoif_sft.json', 'w') as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)
"

# 注册数据集
python -c "
import json, os
p = 'LlamaFactory/data/dataset_info.json'
if os.path.exists(p):
    with open(p) as f: info = json.load(f)
else: info = {}
info['autoif_sft'] = {'file_name': 'autoif_sft.json', 'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}}
info['autoif_dpo'] = {'file_name': 'autoif_dpo.json', 'ranking': True, 'columns': {'prompt': 'instruction', 'query': 'input', 'chosen': 'chosen', 'rejected': 'rejected'}}
with open(p, 'w') as f: json.dump(info, f, ensure_ascii=False, indent=2)
"

cd LlamaFactory
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$STUDENT_PATH" \
    --dataset autoif_sft \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_target q_proj,v_proj \
    --output_dir ../models/model_c_sft \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --fp16 \
    --cutoff_len 2048 \
    --report_to none \
    2>&1 | tee ../logs/sft_train.log
cd ..

echo "✅ SFT 训练完成"

# ================================================================
# 阶段5: DPO 训练
# ================================================================
echo ""
echo "============================================"
echo "  阶段5: DPO 训练"
echo "============================================"

# 准备 DPO 数据
python -c "
import json
dpo_data = []
with open('output/dpo_pairs.jsonl') as f:
    for line in f:
        item = json.loads(line)
        dpo_data.append({'instruction': item.get('instruction',''), 'input': item.get('query',''), 'chosen': item.get('chosen',''), 'rejected': item.get('rejected','')})
print(f'DPO 数据: {len(dpo_data)} 对')
with open('LlamaFactory/data/autoif_dpo.json', 'w') as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)
"

cd LlamaFactory
llamafactory-cli train \
    --stage dpo \
    --do_train \
    --model_name_or_path "$STUDENT_PATH" \
    --adapter_name_or_path ../models/model_c_sft \
    --dataset autoif_dpo \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_target q_proj,v_proj \
    --output_dir ../models/model_c_dpo \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --fp16 \
    --cutoff_len 2048 \
    --pref_beta 0.1 \
    --report_to none \
    2>&1 | tee ../logs/dpo_train.log
cd ..

echo "✅ DPO 训练完成"

# ================================================================
# 阶段6: LoRA 合并
# ================================================================
echo ""
echo "============================================"
echo "  阶段6: LoRA 权重合并"
echo "============================================"

cd LlamaFactory
llamafactory-cli export \
    --model_name_or_path "$STUDENT_PATH" \
    --adapter_name_or_path "../models/model_c_dpo" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "../models/model_merged" \
    --export_size 2 \
    --export_legacy_format false 2>&1 | tee ../logs/merge.log
cd ..

echo "✅ LoRA 合并完成"

# ================================================================
# 阶段7: GPTQ 量化
# ================================================================
echo ""
echo "============================================"
echo "  阶段7: GPTQ 量化（INT4）"
echo "============================================"

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = './models/model_merged'
output_dir = './models/model_gptq_int4'

print('加载合并后的模型...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path, quantize_config=quantize_config,
        trust_remote_code=True, torch_dtype=torch.float16
    )
    
    # 准备校准数据
    calibration_texts = [
        '请用三句话解释什么是机器学习。',
        'Write a Python function that sorts a list.',
        '帮我写一封正式的商务邮件。',
        'Explain the difference between TCP and UDP.',
        '如何在Linux上查看GPU使用情况？',
        '请列举5个常见的数据结构及其特点。',
        'What is the time complexity of binary search?',
        '用简单的语言解释什么是深度学习中的反向传播。',
    ]
    examples = [tokenizer(t, return_tensors='pt') for t in calibration_texts]
    
    print('GPTQ 量化中...')
    model.quantize(examples)
    
    print(f'保存到 {output_dir}...')
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('✅ GPTQ 量化完成!')
except Exception as e:
    print(f'GPTQ 量化失败: {e}')
    print('尝试使用 AutoAWQ...')
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('✅ AWQ 量化完成!')
" 2>&1 | tee logs/quantize.log

echo "✅ 量化完成"

# ================================================================
# 阶段8: vLLM 推理测试
# ================================================================
echo ""
echo "============================================"
echo "  阶段8: 推理测试"
echo "============================================"

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('=== 测试量化模型 ===')
model_path = './models/model_gptq_int4'
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16
    )
    
    test_prompts = [
        '请用3句话解释量子力学。',
        'Write a hello world program in Python.',
    ]
    
    for prompt in test_prompts:
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f'\\n问: {prompt}')
        print(f'答: {response[:200]}')
    
    print('\\n✅ 推理测试通过!')
except Exception as e:
    print(f'推理测试失败: {e}')
" 2>&1 | tee logs/inference_test.log

# ================================================================
# 完成
# ================================================================
echo ""
echo "============================================"
echo "  全流程完成！"
echo "  领域: $DOMAIN"
echo "  结束: $(date)"
echo "============================================"
echo ""
echo "模型文件:"
echo "  SFT LoRA:  models/model_c_sft/"
echo "  DPO LoRA:  models/model_c_dpo/"
echo "  合并模型:  models/model_merged/"
echo "  量化模型:  models/model_gptq_int4/ (如果量化成功)"
echo ""
echo "数据文件:"
echo "  SFT 数据:  output/query_score_filter.jsonl"
echo "  DPO 数据:  output/dpo_pairs.jsonl"
echo ""
echo "日志文件:    logs/"
echo ""
echo "模型大小:"
du -sh models/model_c_sft/ models/model_c_dpo/ models/model_merged/ 2>/dev/null || true

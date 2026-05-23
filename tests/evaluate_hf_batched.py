import torch
import time
import json
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code_sft'))
from utils import compile_eval_func, run_eval_func 

# 1. 配置你要测试的模型字典
models_to_test = {
    "Base-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/student/Qwen/Qwen2.5-1.5B-Instruct",
        "adapter": None
    },
    "SFT-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_sft_merged",
        "adapter": None
    },
    "DPO-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo_merged",
        "adapter": None
    },
    # "GPTQ-Model": {
    #     "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo_merged_gptq_int4",
    #     "adapter": None
    # },
}

TEST_SET_PATH = "output/query_rft_test.jsonl"
OUTPUT_COMPARE_PATH = "output/dpo_alignment_compare_hf_batched.json"

# 设定并发批次大小
BATCH_SIZE = 16

def load_test_data(filepath, limit=200):
    """加载测试数据"""
    data = []
    if not os.path.exists(filepath):
        print(f"❌ 找不到测试集文件: {filepath}")
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if len(data) >= limit: break
    return data

def main():
    test_data = load_test_data(TEST_SET_PATH, limit=200)
    if not test_data: return

    all_responses_history = {}
    if os.path.exists(OUTPUT_COMPARE_PATH):
        with open(OUTPUT_COMPARE_PATH, 'r', encoding='utf-8') as f:
            all_responses_history = json.load(f)
            
    for item in test_data:
        prompt_text = item.get('prompt', '')
        if prompt_text not in all_responses_history:
            all_responses_history[prompt_text] = {}
            
    print(f"✅ 加载了 {len(test_data)} 条测试题，当前 Batch Size: {BATCH_SIZE}")

    for name, config in models_to_test.items():
        base_path = config.get("base")
        adapter_path = config.get("adapter")

        if not os.path.exists(base_path): continue

        print(f"\n{'='*50}")
        print(f"🚀 正在测评 (Transformers 批处理加速): [ {name} ]")
        print(f"{'='*50}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
            
            # 核心配置 1：必须设置左侧 Padding，否则批量生成会全乱套
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                base_path, 
                torch_dtype=torch.float16, # GPTQ 模型建议用 float16
                device_map="auto",
                trust_remote_code=False
            )
            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()

            correct_count = 0
            total_generated_tokens = 0
            start_total_time = time.time()

            # 核心配置 2：将数据按 BATCH_SIZE 分块打包
            for i in range(0, len(test_data), BATCH_SIZE):
                batch_items = test_data[i:i+BATCH_SIZE]
                batch_prompts = [item.get('prompt', '') for item in batch_items]
                
                # 批量格式化 Prompt
                formatted_prompts = []
                for pt in batch_prompts:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions strictly."},
                        {"role": "user", "content": pt}
                    ]
                    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    formatted_prompts.append(formatted)

                # 批量 Tokenize，开启 padding
                inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)
                
                print(f"⏳ 正在推理批次 {i//BATCH_SIZE + 1} (包含 {len(batch_items)} 条数据)...")
                
                # 批量生成
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,        
                    top_p=0.85,              
                    top_k=50,               
                    repetition_penalty=1.05,
                    eos_token_id=[tokenizer.eos_token_id, 151645, 151643],
                    pad_token_id=tokenizer.pad_token_id,
                )

                # 计算输入长度，截断获取真正的生成内容
                input_len = inputs["input_ids"].shape[1]
                
                # ✨ 核心配置 3：批量解码与评判
                for j, generated_sequence in enumerate(outputs):
                    original_item = batch_items[j]
                    prompt_text = original_item.get('prompt', '')
                    eval_funcs_code = original_item.get('eval_func', [])
                    
                    # 提取纯生成的 token 并解码
                    generated_tokens = generated_sequence[input_len:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    # 累加 Token 数用于计算 TPS
                    # 注意：Transformers 的输出包含了 Padding 长度，这里我们只算非 Padding 的真实输出 Token
                    real_token_count = (generated_tokens != tokenizer.pad_token_id).sum().item()
                    total_generated_tokens += real_token_count
                    
                    all_responses_history[prompt_text][name] = response
                    
                    # 验证函数打分
                    is_passed = False
                    for func_code, _ in eval_funcs_code:
                        fn = compile_eval_func(func_code)
                        if fn:
                            res = run_eval_func(fn, response)
                            if res: 
                                is_passed = True
                                break 
                    if is_passed:
                        correct_count += 1

            end_total_time = time.time()
            total_time_taken = end_total_time - start_total_time
            overall_tps = total_generated_tokens / total_time_taken if total_time_taken > 0 else 0

            # 打印战报
            accuracy = (correct_count / len(test_data)) * 100
            print(f"\n📊 {name} 最终战报 (Transformers 批处理版):")
            print(f"  - 指令遵循准确率: {accuracy:.2f}%")
            print(f"  - 总耗时: {total_time_taken:.2f} 秒")
            print(f"  - 整体并发速度: {overall_tps:.2f} tokens/s")
            
            # 清理显存
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 运行失败: {e}")

    # 保存文件
    with open(OUTPUT_COMPARE_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_responses_history, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存所有回答: {OUTPUT_COMPARE_PATH}")

if __name__ == "__main__":
    main()
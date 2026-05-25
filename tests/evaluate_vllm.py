import os
# 在所有导入之前加入这两行
os.environ["VLLM_USE_MODELSCOPE"] = "False"
# 关键：告诉 vLLM 不要尝试去加载那些需要复杂依赖的引导解码器
os.environ["VLLM_NO_OUTLINES"] = "1"
import sys
import time
import json
import numpy as np
from pathlib import Path
import gc
import torch

# ✨ 1. 导入 vLLM 核心组件
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 解开注释：将项目根目录下的 code_sft 文件夹加入系统路径，以便导入 utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code_sft'))
from utils import compile_eval_func, run_eval_func 
# 配置你要测试的模型字典
models_to_test = {
    "Base-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/student/Qwen/Qwen2.5-1.5B-Instruct",
    },
    "SFT-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_sft_merged",
    },
    "DPO-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo_merged",
    },
    "GPTQ-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo_merged_gptq_int4",
    },
}

TEST_SET_PATH = "output/query_rft_test.jsonl"
OUTPUT_COMPARE_PATH = "output/vllm_dpo_alignment_compare.json"

def load_test_data(filepath, limit=200):
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
    if not test_data:
        return

    all_responses_history = {}
    if os.path.exists(OUTPUT_COMPARE_PATH):
        print(f"📂 检测到已存在的对比文件，将进行追加合并：{OUTPUT_COMPARE_PATH}")
        with open(OUTPUT_COMPARE_PATH, 'r', encoding='utf-8') as f:
            all_responses_history = json.load(f)
            
    for item in test_data:
        prompt_text = item.get('prompt', '')
        if prompt_text not in all_responses_history:
            all_responses_history[prompt_text] = {}
            
    print(f"✅ 加载了 {len(test_data)} 条测试题")

    for name, config in models_to_test.items():
        base_path = config.get("base")

        if not os.path.exists(base_path): 
            print(f"⚠️ 找不到基座模型: {base_path}，跳过 {name}")
            continue

        print(f"\n{'='*50}")
        print(f"🚀 正在测评 (vLLM 加速): [ {name} ]")
        print(f"{'='*50}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
            
            # ✨ 2. 批量格式化所有的 Prompt
            formatted_prompts = []
            for item in test_data:
                prompt_text = item.get('prompt', '')
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions strictly."},
                    {"role": "user", "content": prompt_text}
                ]
                # 转换成模型专属的 chat template 字符串
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(formatted_prompt)

            # ✨ 3. 初始化 vLLM 引擎
            # 动态判断是否使用 GPTQ
            quantization_config = "gptq" if "gptq" in base_path.lower() else None
            
            llm = LLM(
                model=base_path, 
                trust_remote_code=True,
                quantization=quantization_config, # 使用动态变量
                enforce_eager=True, 
                dtype="float16",
                gpu_memory_utilization=0.85, 
                max_model_len=2048 
            )

            # ✨ 4. 配置 vLLM 的采样参数 
            sampling_params = SamplingParams(
                temperature=0,                    
                repetition_penalty=1.05,
                max_tokens=512,
                stop_token_ids=[tokenizer.eos_token_id, 151645, 151643]
            )

            print("⏳ 正在进行高并发批量推理，请稍候...")
            # 5. 一键批量生成
            start_time = time.time()
            outputs = llm.generate(formatted_prompts, sampling_params)
            end_time = time.time()

            # 统计与判卷
            correct_count = 0
            total_generated_tokens = 0

            # 遍历 vLLM 返回的批量结果
            for i, output in enumerate(outputs):
                original_item = test_data[i]
                prompt_text = original_item.get('prompt', '')
                eval_funcs_code = original_item.get('eval_func', [])
                
                # 获取生成的文本和 token 数量
                response = output.outputs[0].text.strip()
                total_generated_tokens += len(output.outputs[0].token_ids)
                
                # 保存到历史字典
                all_responses_history[prompt_text][name] = response
                
                # 指标计算：SFT 准确率 (跑验证函数)
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
            
            # ✨ 6. 统一计算总体并发 TPS
            total_time = end_time - start_time
            overall_tps = total_generated_tokens / total_time if total_time > 0 else 0

            # 打印战报
            accuracy = (correct_count / len(test_data)) * 100
            print(f"\n📊 {name} 最终战报:")
            print(f"  - 指令遵循准确率: {accuracy:.2f}%")
            print(f"  - 总体推理总耗时: {total_time:.2f} 秒")
            print(f"  - ⚡ 批量并发速度: {overall_tps:.2f} tokens/s")
            
            # ✨ 7. 暴力清理 vLLM 显存占用 
            del llm
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback; print(f"❌ 运行失败: {e}"); traceback.print_exc()

    # 保存对比数据
    ensure_dir = os.path.dirname(OUTPUT_COMPARE_PATH)
    if not os.path.exists(ensure_dir): os.makedirs(ensure_dir)
    with open(OUTPUT_COMPARE_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_responses_history, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存所有回答: {OUTPUT_COMPARE_PATH}")

if __name__ == "__main__":
    main()

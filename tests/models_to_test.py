import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  
import os
import re

models_to_test = {
    "Base-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/student/Qwen/Qwen2.5-1.5B-Instruct"
    },
    "SFT-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_sft_merged"
    },
    "DPO-Model": {
        "base": "/root/autodl-tmp/AutoIF-LLM/models/model_d_sft_merged",
        "adapter": "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo"
    }
}
# 更有挑战性的指令遵循测试用例
test_cases = [
    ("How do I make my Wi-Fi secure?", 
     "Construct the reply as if it's a telegram. Use 'STOP' at the end of each sentence."),
    ("Explain NLP briefly.",  
     "Your response must be exactly three sentences long."),
    ("What's a good hobby to start?",
    "Answer with words that begin with the letter 'B'."),
    ("Write a short story about a cat.", 
     "Every sentence must start with the letter 'T'."),
]

def generate_response(model, tokenizer, question, instruction):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Follow the user's formatting instructions strictly."},
        {"role": "user", "content": f"{question}\n\nConstraint: {instruction}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,          
        do_sample=True,
        temperature=0.7,            
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=[tokenizer.eos_token_id, 151645, 151643],
        pad_token_id=tokenizer.eos_token_id,
    )
    
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    stop_signals = ["<|im_end|>", "<Task complete"]
    for sig in stop_signals:
        if sig in response:
            response = response.split(sig)[0].strip()
    
    return response

def main():
    for name, config in models_to_test.items():
        base_path = config.get("base")
        adapter_path = config.get("adapter")

        # 检查路径有效性
        if not os.path.exists(base_path):
            print(f"⚠️ 跳过 {name}: 基座路径不存在 {base_path}")
            continue
        if adapter_path and not os.path.exists(adapter_path):
            print(f"⚠️ 跳过 {name}: LoRA路径不存在 {adapter_path}")
            continue

        print(f"\n\n{'='*40}")
        print(f"🚀 正在加载并测评模型: {name}")
        print(f"{'='*40}")
        
        try:
            # 1. 加载 Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
            
            # 2. 加载 基座模型
            print("⏳ 正在加载基座模型权重...")
            model = AutoModelForCausalLM.from_pretrained(
                base_path, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True
            )

            # 3. 如果配置了 adapter_path，则将其挂载到基座模型上
            if adapter_path:
                print(f"🔗 正在挂载 LoRA 权重: {adapter_path.split('/')[-1]}")
                model = PeftModel.from_pretrained(model, adapter_path)

            model.eval()
            print("✅ 模型加载完成，开始推理！")

            for i, (q, ins) in enumerate(test_cases):
                print(f"\n[测试集 {i+1}]")
                print(f"❓ 问题: {q}")
                print(f"📜 约束: {ins}")
                
                response = generate_response(model, tokenizer, q, ins)
                
                print(f"🤖 回答:\n{'-'*20}\n{response}\n{'-'*20}")
            
            # 释放显存，防止 OOM
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 模型 {name} 加载或运行失败: {e}")

if __name__ == "__main__":
    main()
import os
os.environ["OMP_NUM_THREADS"] = "4"
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

model_path = "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo_merged"
sft_data_path = "/root/autodl-tmp/AutoIF-LLM/output/IF_sft_data.json"
output_dir = "/root/autodl-tmp/AutoIF-LLM/models/model_d_dpo_merged_gptq_int4"

def prepare_calibration_data(data_path, num_samples=128):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data = data[:num_samples]
    calibration_texts = []
    for item in data:
        if "instruction" in item:
            text = f"{item['instruction']}\n{item.get('input', '')}"
            calibration_texts.append(text)
    return calibration_texts

# 1. 准备校准数据
calibration_texts = prepare_calibration_data(sft_data_path, num_samples=128)
print(f"✅ 准备了 {len(calibration_texts)} 条校准样本")

# 2. 加载模型和 Tokenizer
print("加载模型中...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"         
)

# 3. 初始化量化器
# 将校准文本直接传给 quantizer
quantizer = GPTQQuantizer(
    bits=4,
    dataset=calibration_texts, 
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
    sym=True
)

# 4. 执行量化 
print("开始 INT4 量化 (这可能需要几分钟)...")
# model = quantizer.quantize(model, tokenizer)
model = quantizer.quantize_model(model, tokenizer)  # 加上 _model
# 5. 保存
print(f"保存量化模型到: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ 量化完成！")
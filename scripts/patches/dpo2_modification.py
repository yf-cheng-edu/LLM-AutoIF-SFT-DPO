import json
import os

# 1. 扁平化数据：从 [{"from": "human", "value": "..."}] 提取出纯文本
input_file = "/root/autodl-tmp/AutoIF-LLM/output/dpo_pairs.jsonl"
output_file = "/root/autodl-tmp/AutoIF-LLM/output/dpo_pairs_flat.jsonl"

print("正在扁平化 DPO 数据...")
with open(input_file, 'r', encoding='utf-8') as f, \
     open(output_file, 'w', encoding='utf-8') as out:
    for line in f:
        data = json.loads(line)
        # 提取 human 的提问字符串
        prompt_text = data['conversations'][0]['value']
        new_item = {
            "prompt": prompt_text,
            "chosen": data['chosen'],
            "rejected": data['rejected']
        }
        out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

# 2. 更新 LlamaFactory 配置
config_path = "/root/autodl-tmp/AutoIF-LLM/LlamaFactory/data/dataset_info.json"
new_config = {
    "file_name": "../../output/dpo_pairs_flat.jsonl",
    "ranking": True,
    "columns": {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}

with open(config_path, 'r', encoding='utf-8') as f:
    content = json.load(f)

content["autoif_dpo"] = new_config

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(content, f, indent=2, ensure_ascii=False)

print("✅ 数据扁平化完成，并已成功注册为 alpaca 格式的 autoif_dpo！")
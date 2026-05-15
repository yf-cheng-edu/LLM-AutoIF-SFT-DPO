import json
import os

path = "/root/autodl-tmp/AutoIF-LLM/LlamaFactory/data/dataset_info.json"

# 修正后的配置：必须包含 "ranking": true
correct_dpo_config = {
    "file_name": "../../output/dpo_pairs.jsonl",
    "formatting": "sharegpt",
    "ranking": True,  # 关键点：告诉框架这是偏好排序数据
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}

if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    
    # 覆盖配置
    content["autoif_dpo"] = correct_dpo_config
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    print("✅ DPO 配置已修正（添加了 ranking 标志）！")
else:
    print("❌ 找不到配置文件")
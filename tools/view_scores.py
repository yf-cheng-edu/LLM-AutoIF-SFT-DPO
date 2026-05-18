import json
import re
import os
from collections import defaultdict

# 定义输入文件路径
input_file = "../output/query_rft_score.jsonl"
output_dir = "../output/score_segments"

if not os.path.exists(input_file):
    print(f"找不到文件: {input_file}，请检查路径。")
    exit()

os.makedirs(output_dir, exist_ok=True)
score_buckets = defaultdict(list)

print("正在读取并解析评分数据...\n")

# 1. 读取并按分数分类
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        
        # 获取 LLM 的评分回复
        gen_text = data.get('gen', [''])[0] 
        
        # 使用正则提取 "Score: X" 中的数字
        match = re.search(r'Score:\s*(\d+)', gen_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
        else:
            score = -1 # 解析失败的归为 -1 类
            
        score_buckets[score].append(data)

# 2. 打印分布统计
print("📊 === 评分段数据分布统计 ===")
total_samples = 0
for score in sorted(score_buckets.keys(), reverse=True):
    count = len(score_buckets[score])
    total_samples += count
    score_label = f"{score} 分" if score >= 0 else "解析失败"
    print(f" {score_label.rjust(6)} : {count} 条")
print(f"--------------------------\n 总计 : {total_samples} 条\n")

# 3. 分别保存到不同的文件
print("💾 正在将各分数段数据拆分保存...")
for score, items in score_buckets.items():
    score_label = str(score) if score >= 0 else "error"
    out_file = os.path.join(output_dir, f"score_{score_label}.jsonl")
    
    with open(out_file, 'w', encoding='utf-8') as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n✅ 完成！所有拆分后的文件已保存在: {output_dir}/ 目录下")
print("你可以直接打开对应的文件来查看具体内容。")
import json
import os
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区 =================
client = OpenAI(
    api_key="YOUR_API_KEY",  
    base_url="https://api.deepseek.com/v1" 
)
JUDGE_MODEL = "deepseek-chat" 

# INPUT_FILE = "output/dpo_alignment_compare_hf_batched.json"
INPUT_FILE = "output/vllm_dpo_alignment_compare.json"

OUTPUT_FILE = "output/all_models_judge_results.json"

# 定义你要进行两两 PK 的组合

PAIRS_TO_COMPARE = [
    ("Base-Model", "SFT-Model"),
    ("Base-Model", "DPO-Model"),
    ("SFT-Model", "DPO-Model"),
    ("DPO-Model", "GPTQ-Model"),
    ("SFT-Model", "GPTQ-Model"),
    ("Base-Model", "GPTQ-Model"),
]


def get_judge_result(prompt, ans_a, ans_b):
    sys_prompt = """You are an expert that is good at judging whether a response is following the instruction and query.
Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.

Your task is to compare two AI assistants (Assistant A and Assistant B) and judge which one provides a better response.
Criteria:
1. Strictness: First and foremost, check which response better strictly follows the formatting/constraints in the instruction.
2. Helpfulness: If both follow the instruction equally well (or both fail), judge which response is more helpful and highly related to the query.

You must strictly return a JSON object with two keys:
- "analysis": A detailed step-by-step reasoning.
- "winner": Must be exactly "A", "B", or "Tie"."""

    user_prompt = f"""[Prompt (Instruction & Query)]
{prompt}

[Assistant A]
{ans_a}

[Assistant B]
{ans_b}

Please judge which assistant is better and output in JSON format."""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API 请求失败: {e}")
        return {"winner": "Error", "analysis": str(e)}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化成绩单统计字典
    stats = {f"{model_a}_vs_{model_b}": {"A_wins": 0, "B_wins": 0, "Ties": 0} 
             for model_a, model_b in PAIRS_TO_COMPARE}
    
    results = []
    
    print(f"✅ 成功读取 {len(data)} 条对比数据，开始多轮 AI 裁判打分...")

    # 遍历所有测试题
    for prompt, models_responses in tqdm(data.items(), desc="全面评测中"):
        current_item_result = {
            "prompt": prompt,
            "responses": models_responses,
            "comparisons": {}
        }
        
        # 遍历配置好的对决组合
        for model_a, model_b in PAIRS_TO_COMPARE:
            ans_a = models_responses.get(model_a, "")
            ans_b = models_responses.get(model_b, "")
            pair_key = f"{model_a}_vs_{model_b}"
            
            # 过滤异常或重复数据
            if not ans_a or not ans_b or ans_a == ans_b:
                stats[pair_key]["Ties"] += 1
                current_item_result["comparisons"][pair_key] = {"winner": "Tie", "analysis": "Identical or empty responses."}
                continue
                
            # 调用裁判 API
            judge = get_judge_result(prompt, ans_a, ans_b)
            winner = judge.get("winner", "")
            
            # 记录战绩
            if winner == "A":
                stats[pair_key]["A_wins"] += 1
            elif winner == "B":
                stats[pair_key]["B_wins"] += 1
            else:
                stats[pair_key]["Ties"] += 1
                
            current_item_result["comparisons"][pair_key] = judge
            
        results.append(current_item_result)

    # 保存完整结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印最终战报
    print("\n" + "="*50)
    print("🏆 LLM-as-a-Judge 进化之路完整战报")
    print("="*50)
    
    for pair_key, record in stats.items():
        model_a, model_b = pair_key.split("_vs_")
        total = record["A_wins"] + record["B_wins"] + record["Ties"]
        if total == 0: continue
        
        print(f"\n🥊 【{model_a}】 VS 【{model_b}】")
        print(f"  - {model_a} 胜率: {(record['A_wins']/total)*100:.2f}%")
        print(f"  - {model_b} 胜率: {(record['B_wins']/total)*100:.2f}%")
        print(f"  - 平局率 (Tie) : {(record['Ties']/total)*100:.2f}%")
        
    print(f"\n💾 详细判卷理由已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
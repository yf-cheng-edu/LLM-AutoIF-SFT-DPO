"""
DPO 步骤 2: DPO 偏好对构建
根据响应评分构建正负样本对，用于 DPO 训练

核心逻辑:
  正样本: 验证函数评分 >= 0.5 的响应 (遵循了指令格式)
  负样本: 验证函数评分 == 0 的响应 (完全未遵循指令格式)
  组合方式: 每条指令取 1~2 个正样本 × 1~2 个负样本 → 偏好对
  输出格式: LLaMA Factory DPO 格式 (conversations + chosen + rejected)

输入: output/dpo_eval_score_results.jsonl
输出: output/dpo_pairs.jsonl
"""
import json
import random
import itertools
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code_sft'))
from utils import (
    logger, check_input_file, ensure_output_dir,
    DPO_EVAL_SCORE_PATH, DPO_PAIRS_PATH, DPO_POSITIVE_THRESHOLD
)


def random_sample(lst: list, n: int) -> list:
    """随机采样 n 个元素，不足则返回全部"""
    return random.sample(lst, min(n, len(lst)))


def main():
    check_input_file(DPO_EVAL_SCORE_PATH, "DPO步骤2-偏好对构建")
    ensure_output_dir()

    total_pairs = 0
    skip_count = 0

    with open(DPO_EVAL_SCORE_PATH, 'r', encoding='utf-8') as infile, \
         open(DPO_PAIRS_PATH, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            instruction = data.get('query', '')
            if "[instruction]" in instruction.lower():
                import re
                match = re.search(r'\[instruction\]\s*(.+?)(?:\n|$)', instruction, re.IGNORECASE)
                if match:
                    instruction = match.group(1).strip()

            # 分离正负样本 (关键修复: 使用 and 确保两边都有样本)
            positive_cases = [
                case[0] for case in data.get("response", [])
                if case[1] >= DPO_POSITIVE_THRESHOLD
            ]
            negative_cases = [
                case[0] for case in data.get("response", [])
                if case[1] == 0
            ]

            # 需要同时有正样本和负样本
            if len(positive_cases) >= 2 and len(negative_cases) >= 2:
                positive_samples = random_sample(positive_cases, 2)
                negative_samples = random_sample(negative_cases, 2)
            elif len(positive_cases) >= 1 and len(negative_cases) >= 1:
                positive_samples = random_sample(positive_cases, 1)
                negative_samples = random_sample(negative_cases, 1)
            else:
                skip_count += 1
                continue

            # 生成所有正负组合
            combinations = set(itertools.product(positive_samples, negative_samples))

            for pos, neg in combinations:
                pair = {
                    "conversations": [
                        {"from": "human", "value": instruction}
                    ],
                    "chosen": pos,
                    "rejected": neg
                }
                outfile.write(json.dumps(pair, ensure_ascii=False) + '\n')
                total_pairs += 1

    logger.info(f"✅ DPO 数据构建完成! 共 {total_pairs} 个偏好对 (跳过 {skip_count} 条)")
    logger.info(f"   保存位置: {DPO_PAIRS_PATH}")


if __name__ == "__main__":
    main()

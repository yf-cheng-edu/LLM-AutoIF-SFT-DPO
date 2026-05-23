import json
import numpy as np
import sys
import random
from pathlib import Path
from tqdm import tqdm

# 确保能引到你的 utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code_sft'))

from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    compile_eval_func, run_eval_func, QUERY_RFT_PATH
)

def main():
    check_input_file(QUERY_RFT_PATH, "提取测试集")

    # 1. 读取原始生成的数据
    results = read_jsonl(QUERY_RFT_PATH)
    logger.info(f"读取原始数据集: {QUERY_RFT_PATH}，共 {len(results)} 条")

    # 2. 随机打乱数据，确保抽取的 200 条覆盖不同类型的指令
    random.seed(42) # 固定随机种子，保证每次运行抽取的结果一致
    random.shuffle(results)

    test_samples = []
    TARGET_TEST_SIZE = 200

    for result in tqdm(results, desc="筛选高质量测试集"):
        if len(test_samples) >= TARGET_TEST_SIZE:
            logger.info("已成功收集 200 条高质量数据，提前结束遍历。")
            break
            
        # 提取并编译这道题的验证函数
        eval_funcs = []
        for func_code, score in result.get('eval_func', []):
            fn = compile_eval_func(func_code)
            if fn is not None:
                eval_funcs.append(fn)
        
        if not eval_funcs:
            continue

        # 评估这道题下的所有 gpt-answer
        is_high_quality = False
        for response in result.get('gpt-answer', []):
            acc = []
            for eval_fn in eval_funcs:
                res = run_eval_func(eval_fn, response)
                if res is not None:
                    try:
                        acc.append(int(bool(res)))
                    except (ValueError, TypeError):
                        continue
            
            # 计算当前这个回答的通过率
            accuracy = float(np.mean(acc)) if acc else 0.0
            
            # 只要有 1 个回答完美通过了验证函数（得分为1.0）
            if accuracy >= 1.0:
                is_high_quality = True
                break # 找到合格的回答，不再校验这道题的其他 answer
        
        # 3. 将符合条件的高质量数据加入测试集
        if is_high_quality:
            test_samples.append(result)

    # 4. 仅保存测试集文件
    TEST_OUTPUT_PATH = "output/query_rft_test.jsonl"
    write_jsonl(TEST_OUTPUT_PATH, test_samples)
    
    logger.info(f"✅ 提取完成！")
    logger.info(f"  - 🎯 成功获得高质量测试集: {len(test_samples)} 条 -> 保存至 {TEST_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
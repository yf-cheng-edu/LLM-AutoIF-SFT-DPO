"""
步骤 2: 验证函数与测试用例生成
使用监督模型为每个指令生成可执行的 Python 验证函数和测试用例

输入: sample_data/seed_instruction.txt + output/augment_instructions.txt
输出: output/eval_func_rft.jsonl (包含 gpt-answer 字段)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, call_llm_batch, ensure_output_dir, check_input_file, write_jsonl,
    SEED_INSTRUCTION_PATH, AUGMENT_INSTRUCTIONS_PATH, EVAL_FUNC_RFT_PATH,
    K_VERIFICATION
)

# 验证函数生成提示词
# 要求教师模型为每条指令生成 Python evaluate() 函数 + 3个测试用例
# 输出格式: JSON {"func": "def evaluate(response):...", "cases": [{"input":..., "output":true/false}]}
PROMPT_TEMPLATE = (
    "You are an expert for writing evaluation functions in Python to evaluate "
    "whether a response strictly follows an instruction.\n"
    "Here is the instruction: {instruction}\n"
    "Please write a Python function named `evaluate` to evaluate whether an "
    "input string `response` follows this instruction. If it follows, simply "
    "return True, otherwise return False.\n"
    "Please response with a single JSON includes the evaluation function in "
    "the key `func`, and a list of three test cases in the key `cases`, which "
    "includes an input in the key `input` and an expected output in the key "
    "`output` (true or false).\n"
    'Here is an example of output JSON format:\n'
    '```json\n'
    '{{"func": "def evaluate(response):\\n    return len(response.split()) <= 50", '
    '"cases": [{{"input": "Short response.", "output": true}}, '
    '{{"input": "", "output": true}}, '
    '{{"input": "A very very long response that exceeds the word limit...", "output": false}}]}}\n'
    '```'
)


def main():
    check_input_file(SEED_INSTRUCTION_PATH, "步骤2-验证函数生成")
    check_input_file(AUGMENT_INSTRUCTIONS_PATH, "步骤2-验证函数生成")
    ensure_output_dir()

    # 读取所有指令
    with open(SEED_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
        seed_instructions = [line.strip() for line in f if line.strip()]

    with open(AUGMENT_INSTRUCTIONS_PATH, 'r', encoding='utf-8') as f:
        augment_instructions = [line.strip() for line in f if line.strip()]

    all_instructions = seed_instructions + augment_instructions
    logger.info(f"总指令数: {len(all_instructions)} "
                f"(种子: {len(seed_instructions)}, 增强: {len(augment_instructions)})")

    # 构造提示词
    prompts = [PROMPT_TEMPLATE.format(instruction=inst) for inst in all_instructions]

    # 批量调用 LLM 生成验证函数 (每个指令生成 K 个)
    logger.info(f"为每个指令生成 {K_VERIFICATION} 个验证函数...")
    responses = call_llm_batch(
        prompts,
        system_prompt="You are an expert Python programmer specializing in writing evaluation functions.",
        temperature=0.8,
        max_tokens=2048,
        n=K_VERIFICATION,
        desc="生成验证函数"
    )

    # 组装输出数据
    outputs = []
    for instruction, prompt, resp_list in zip(all_instructions, prompts, responses):
        if resp_list:
            outputs.append({
                "instruction": instruction,
                "prompt": prompt,
                "gpt-answer": resp_list  # K 个 LLM 响应
            })

    write_jsonl(EVAL_FUNC_RFT_PATH, outputs)
    logger.info(f"✅ 验证函数生成完成! {len(outputs)}/{len(all_instructions)} 条指令成功")


if __name__ == "__main__":
    main()

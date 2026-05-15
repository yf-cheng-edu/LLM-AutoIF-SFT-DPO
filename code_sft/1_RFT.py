"""
步骤 1: 指令增强 (RFT - Rejection Fine-Tuning Sampling)
使用监督模型从种子指令生成大量增强指令

输入: sample_data/seed_instruction.txt
输出: output/augment_instructions.txt
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, call_llm, ensure_output_dir, check_input_file,
    SEED_INSTRUCTION_PATH, AUGMENT_INSTRUCTIONS_PATH, K_AUGMENT
)

random.seed(0)

# 指令增强提示词模板
# 要求教师模型基于种子指令生成50条新的格式约束指令
# 关键约束: 只要格式类指令(可用Python验证)，不要风格类指令(如隐喻、翻译)
AUGMENT_PROMPT = """You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
- Instructions are about the format but not style of a response
- Whether instructions are followed can be easily evaluate by a Python function
Here are some examples of instructions we need:
{seed_instructions}
Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we do not need:
- Incorporate a famous historical quote seamlessly into your answer
- Translate your answer into Pig Latin
- Use only words that are also a type of food
- Respond with a metaphor in every sentence
- Write the response as if you are a character from a Shakespearean play
Please generate one instruction per line in your response and start each line with '- '."""


def main():
    # 检查输入文件
    check_input_file(SEED_INSTRUCTION_PATH, "步骤1-指令增强")
    ensure_output_dir()

    # 读取种子指令
    with open(SEED_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
        seed_instructions = [line.strip() for line in f if line.strip()]
    logger.info(f"读取 {len(seed_instructions)} 条种子指令")

    # 构造提示词
    seed_text = '\n'.join(
        [f"- {inst}" if not inst.startswith('-') else inst for inst in seed_instructions]
    )
    prompt = AUGMENT_PROMPT.format(seed_instructions=seed_text)

    # 调用 LLM 生成增强指令 (调用 K 轮获得更多多样性)
    logger.info(f"开始调用 LLM 生成增强指令 (共 {K_AUGMENT} 轮)...")
    all_instructions = set()

    for i in range(K_AUGMENT):
        logger.info(f"第 {i+1}/{K_AUGMENT} 轮生成中...")
        try:
            responses = call_llm(prompt, temperature=0.9, max_tokens=2048)
            for response in responses:
                for line in response.split('\n'):
                    line = line.strip()
                    # 解析 "- xxx" 或 "• xxx" 格式
                    if line.startswith('- '):
                        instruction = line[2:].strip()
                    elif line.startswith('• '):
                        instruction = line[2:].strip()
                    elif line and line[0].isdigit() and '.' in line[:4]:
                        # "1. xxx" 格式
                        instruction = line.split('.', 1)[1].strip()
                    else:
                        continue
                    if len(instruction) > 10:
                        all_instructions.add(instruction)
        except Exception as e:
            logger.error(f"第 {i+1} 轮生成失败: {e}")

    # 随机打乱
    augmented = list(all_instructions)
    random.shuffle(augmented)

    # 保存增强指令
    with open(AUGMENT_INSTRUCTIONS_PATH, 'w', encoding='utf-8') as f:
        for inst in augmented:
            f.write(inst + '\n')

    logger.info(f"✅ 指令增强完成!")
    logger.info(f"   种子指令: {len(seed_instructions)} 条")
    logger.info(f"   新增增强指令: {len(augmented)} 条")
    logger.info(f"   保存位置: {AUGMENT_INSTRUCTIONS_PATH}")


if __name__ == "__main__":
    main()

"""
步骤 6: 查询增强与响应生成
将指令与真实查询配对，并使用监督模型生成多个响应

输入: output/back_trans_filter.jsonl + ShareGPT 数据集 (可选)
输出: output/query_rft.jsonl (包含 gpt-answer 字段)

ShareGPT 数据路径可通过环境变量设置:
    export SHAREGPT_PATH="./data/sharegpt.json"
"""
import json
import random
import copy
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    call_llm, call_llm_batch,
    BACK_TRANS_FILTER_PATH, QUERY_RFT_PATH,
    QUERIES_PER_INSTRUCTION, K_RESPONSE
)

random.seed(0)

# ShareGPT 数据路径
SHAREGPT_PATH = os.environ.get("SHAREGPT_PATH", "./data/sharegpt.json")


def load_queries_from_sharegpt(path: str) -> list:
    """从 ShareGPT 数据集加载用户查询"""
    queries = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]

        for item in data:
            # 支持 ShareGPT conversations 格式
            if 'conversations' in item:
                for msg in item['conversations']:
                    if msg.get('from') in ('human', 'user') and msg.get('value'):
                        q = msg['value'].strip()
                        if 20 < len(q) < 300:
                            queries.append(q)
                        break
            # 支持 messages 格式
            elif 'messages' in item:
                for msg in item['messages']:
                    if msg.get('role') in ('human', 'user') and msg.get('content'):
                        q = msg['content'].strip()
                        if 20 < len(q) < 300:
                            queries.append(q)
                        break

        logger.info(f"从 ShareGPT 加载 {len(queries)} 条查询")
    except FileNotFoundError:
        logger.warning(f"ShareGPT 数据集未找到: {path}")
    except Exception as e:
        logger.warning(f"加载 ShareGPT 失败: {e}")

    return queries


def generate_queries_with_llm(count: int = 500) -> list:
    """使用 LLM 生成查询 (当没有 ShareGPT 数据时的备用方案)"""
    logger.info(f"使用 LLM 生成 {count} 条查询...")

    prompt = f"""Please generate {count} diverse, realistic user queries that people might ask an AI assistant.
Requirements:
- Each query should be 20-200 characters long
- Queries should cover diverse topics (science, daily life, work, education, technology, etc.)
- Queries should be natural and realistic
- One query per line, do not number them
Generate exactly {count} queries:"""

    queries = []
    # 分批生成以获得更多多样性
    batch_size = min(100, count)
    num_batches = (count + batch_size - 1) // batch_size

    for i in range(num_batches):
        try:
            responses = call_llm(prompt, temperature=0.95, max_tokens=2048)
            for resp in responses:
                for line in resp.split('\n'):
                    line = line.strip().lstrip('0123456789.-) •')
                    if 20 < len(line) < 300:
                        queries.append(line)
        except Exception as e:
            logger.error(f"LLM 生成查询失败 (第 {i+1} 批): {e}")

    queries = list(set(queries))  # 去重
    logger.info(f"LLM 共生成 {len(queries)} 条不重复查询")
    return queries


def main():
    check_input_file(BACK_TRANS_FILTER_PATH, "步骤6-查询增强")

    # 读取过滤后的指令数据
    filter_results = read_jsonl(BACK_TRANS_FILTER_PATH)
    logger.info(f"读取 {len(filter_results)} 条过滤后的指令")

    # 加载查询数据
    queries = load_queries_from_sharegpt(SHAREGPT_PATH)
    if len(queries) < QUERIES_PER_INSTRUCTION:
        logger.warning(f"查询数据不足 (当前 {len(queries)} 条, 需要至少 {QUERIES_PER_INSTRUCTION} 条)")
        logger.info("使用 LLM 生成补充查询...")
        extra = generate_queries_with_llm(count=max(500, QUERIES_PER_INSTRUCTION * 5))
        queries.extend(extra)
        queries = list(set(queries))

    if len(queries) < QUERIES_PER_INSTRUCTION:
        logger.error(f"查询数据严重不足 ({len(queries)} 条), 无法继续。"
                     f"请提供 ShareGPT 数据: export SHAREGPT_PATH=你的路径")
        return

    logger.info(f"可用查询数: {len(queries)}")

    # ========== 阶段1: 构造查询-指令配对 ==========
    logger.info("阶段1: 构造查询-指令配对...")
    inputs = []
    for instruction_data in tqdm(filter_results, desc="构造配对"):
        sample_size = min(QUERIES_PER_INSTRUCTION, len(queries))
        ins_queries = random.sample(queries, sample_size)

        for q in ins_queries:
            prompt = (
                f"Please answer the query strictly following the instruction.\n"
                f"[instruction] {instruction_data['instruction']}\n"
                f"[Query] {q}"
            )
            item = copy.deepcopy(instruction_data)
            item['prompt'] = prompt
            item['query'] = q
            inputs.append(item)

    logger.info(f"共生成 {len(inputs)} 个查询-指令配对")

    # ========== 阶段2: 使用监督模型生成 K 个响应 ==========
    logger.info(f"阶段2: 为每个查询生成 {K_RESPONSE} 个响应...")
    prompts = [item['prompt'] for item in inputs]

    responses = call_llm_batch(
        prompts,
        system_prompt="You are a helpful assistant. Follow the instruction precisely when answering the query.",
        temperature=0.8,
        max_tokens=2048,
        n=K_RESPONSE,
        desc="生成响应"
    )

    # 组装输出
    outputs = []
    for item, resp_list in zip(inputs, responses):
        if resp_list:
            item['gpt-answer'] = resp_list
            outputs.append(item)

    write_jsonl(QUERY_RFT_PATH, outputs)
    logger.info(f"✅ 查询增强完成! 成功生成 {len(outputs)} 条数据")


if __name__ == "__main__":
    main()

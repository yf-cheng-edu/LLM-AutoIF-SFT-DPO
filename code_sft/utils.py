"""
AutoIF Pipeline 工具模块
提供统一的 LLM 调用接口、路径管理和通用工具函数

使用前请设置环境变量 (在 AutoDL 上使用 vLLM 部署教师模型):
    export SUPERVISOR_API_BASE="https://api.deepseek.com/v1"   # vLLM 服务地址
    export SUPERVISOR_API_KEY="Your_API_KEY"                   # API Key
    export SUPERVISOR_MODEL="deepseek-chat"                    # 模型名称
"""

import os
import json
import re
import time
import signal
import logging
from typing import List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("autoif")

# ========== 路径配置 ==========
OUTPUT_DIR = "./output"
SAMPLE_DIR = "./sample_data"

# 输入文件
SEED_INSTRUCTION_PATH = os.path.join(SAMPLE_DIR, "seed_instruction.txt")

# 各步骤输出路径 (按执行顺序)
AUGMENT_INSTRUCTIONS_PATH = os.path.join(OUTPUT_DIR, "augment_instructions.txt")
EVAL_FUNC_RFT_PATH = os.path.join(OUTPUT_DIR, "eval_func_rft.jsonl")
CROSS_VALIDATION_PATH = os.path.join(OUTPUT_DIR, "cross_validation.jsonl")
BACK_TRANS_PATH = os.path.join(OUTPUT_DIR, "back_trans.jsonl")
BACK_TRANS_FILTER_PATH = os.path.join(OUTPUT_DIR, "back_trans_filter.jsonl")
QUERY_RFT_PATH = os.path.join(OUTPUT_DIR, "query_rft.jsonl")
QUERY_RFT_SCORE_PATH = os.path.join(OUTPUT_DIR, "query_rft_score.jsonl")
QUERY_SCORE_FILTER_PATH = os.path.join(OUTPUT_DIR, "query_score_filter.jsonl")
SFT_DATA_PATH = os.path.join(OUTPUT_DIR, "IF_sft_data.json")
DPO_EVAL_SCORE_PATH = os.path.join(OUTPUT_DIR, "dpo_eval_score_results.jsonl")
DPO_PAIRS_PATH = os.path.join(OUTPUT_DIR, "dpo_pairs.jsonl")

# ========== 超参数默认值 ==========
# 1. 增加指令增强的轮数
K_AUGMENT = 18                  # 指令增强调用次数 (每次生成 ~50 条)
K_VERIFICATION = 5              # 每条指令生成的验证函数数
K_RESPONSE = 5                  # 每个查询生成的响应数
CROSS_VAL_THRESHOLD = 0.8       # 交叉验证准确率阈值
MIN_EVAL_FUNCS = 3              # 最少验证函数数
MIN_TEST_CASES = 10             # 最少测试用例数
QUALITY_SCORE_THRESHOLD = 9     # 质量评分阈值 (0-10)
QUERIES_PER_INSTRUCTION = 10    # 每条指令配对的查询数
DPO_POSITIVE_THRESHOLD = 0.5    # DPO 正样本阈值
EXEC_TIMEOUT = 5                # 验证函数执行超时 (秒)
LLM_MAX_WORKERS = 30            # LLM 并发调用数


# ========== 文件操作 ==========

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_input_file(path: str, step_name: str):
    """检查输入文件是否存在"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"[{step_name}] 输入文件不存在: {path}\n"
            f"请先运行上一步生成该文件。\n"
            f"{'='*60}"
        )


def write_jsonl(path: str, data: list):
    """写入 JSONL 文件"""
    ensure_output_dir()
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"已写入 {len(data)} 条数据到 {path}")


def read_jsonl(path: str) -> list:
    """读取 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"从 {path} 读取 {len(data)} 条数据")
    return data


# ========== LLM 调用 ==========
def get_llm_client():
    """获取 OpenAI 兼容的 LLM 客户端 (支持 vLLM 本地服务)"""
    # 为了线程安全，每次调用创建独立 client
    from openai import OpenAI
    api_base = os.environ.get("SUPERVISOR_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("SUPERVISOR_API_KEY", "EMPTY")
    client = OpenAI(base_url=api_base, api_key=api_key)
    return client

def get_model_name():
    """获取监督模型名称"""
    return os.environ.get("SUPERVISOR_MODEL", "default")

def call_llm(prompt: str, system_prompt: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 2048,
             n: int = 1) -> List[str]:
    """
    调用 LLM 生成响应
    """
    client = get_llm_client()
    model = get_model_name()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    api_base_lower = str(client.base_url).lower()
    if n > 1 and ("deepseek" in api_base_lower or "deepseek" in model.lower()):
        # 如果需要生成 5 个响应 (n=5)，我们不发给服务器 n=5 的请求
        # 而是直接在这里循环 5 次，每次请求 n=1
        results = []
        for _ in range(n):
            r = call_llm(prompt, system_prompt, temperature, max_tokens, n=1)
            results.extend(r)
        return results
    # =====================================================================

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            err_msg = str(e).lower()
            # 兼容其他不支持 n>1 的模型的被动回退逻辑
            if n > 1 and ("n" in err_msg or "not support" in err_msg):
                logger.info(f"模型不支持 n={n}，回退为 {n} 次单独调用")
                results = []
                for _ in range(n):
                    r = call_llm(prompt, system_prompt, temperature, max_tokens, n=1)
                    results.extend(r)
                return results

            logger.warning(f"LLM 调用失败 (第 {attempt+1}/{max_retries} 次): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM 调用失败，已重试 {max_retries} 次")

def call_llm_batch(prompts: List[str], system_prompt: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 2048,
                   n: int = 1, max_workers: int = None,
                   desc: str = "LLM 批量调用") -> List[List[str]]:
    """
    批量调用 LLM (多线程并发)

    Args:
        prompts: 提示词列表
        system_prompt: 系统提示词 (可选)
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        n: 每个提示词生成的响应数
        max_workers: 最大并发数 (默认使用 LLM_MAX_WORKERS)
        desc: 进度条描述

    Returns:
        响应列表，results[i] 是第 i 个提示词的 n 条响应
    """
    if max_workers is None:
        max_workers = LLM_MAX_WORKERS

    results = [None] * len(prompts)
    failed_indices = []

    def _call_single(idx: int, prompt: str):
        return idx, call_llm(prompt, system_prompt, temperature, max_tokens, n)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_single, i, p): i for i, p in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(prompts), desc=desc):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                idx = futures[future]
                logger.error(f"第 {idx} 个提示词调用失败: {e}")
                results[idx] = []
                failed_indices.append(idx)

    if failed_indices:
        logger.warning(f"共有 {len(failed_indices)}/{len(prompts)} 个调用失败")

    return results


# ========== 验证函数工具 ==========

def compile_eval_func(func_code: str) -> Optional[Callable]:
    """
    编译验证函数代码，返回可调用的 evaluate 函数对象

    Args:
        func_code: Python 函数代码字符串

    Returns:
        evaluate 函数对象，或 None (如果编译失败)
    """
    # 处理转义字符
    if '\\n' in func_code:
        func_code = func_code.replace('\\n', '\n')
    if '\\t' in func_code:
        func_code = func_code.replace('\\t', '\t')
    func_code = func_code.strip()

    # 过滤危险代码行
    safe_lines = []
    for line in func_code.split('\n'):
        line_lower = line.strip().lower()
        if any(kw in line_lower for kw in [
            'os.system', 'subprocess', 'shutil.rmtree', '__import__',
            'download', 'requests.', 'urllib', 'socket.', 'os.remove'
        ]):
            continue
        safe_lines.append(line)
    func_code = '\n'.join(safe_lines)

    local_vars = {}
    try:
        exec(func_code, {"__builtins__": __builtins__}, local_vars)
    except Exception:
        return None

    return local_vars.get('evaluate', None)


def run_eval_func(eval_func: Callable, input_str: str, timeout: int = None) -> Any:
    """
    运行验证函数 (带超时保护)

    - Linux: 使用 signal.SIGALRM
    - Windows: 使用线程超时控制
    """
    if timeout is None:
        timeout = EXEC_TIMEOUT

    # Windows 环境不支持 SIGALRM
    if os.name == "nt":
        import threading

        result = [None]
        error = [None]

        def _target():
            try:
                result[0] = eval_func(input_str)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return None
        if error[0] is not None:
            return None
        return result[0]

    def timeout_handler(signum, frame):
        raise TimeoutError("函数执行超时")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        result = eval_func(input_str)
    except Exception:
        result = None
    finally:
        signal.alarm(0)

    return result


def parse_json_from_response(response: str) -> Optional[dict]:
    """
    从 LLM 响应中解析 JSON
    按优先级尝试: ```json``` 代码块 → ``` 代码块 → 直接解析 → 正则提取
    """
    # 1. 尝试从 ```json...``` 代码块中提取
    json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_blocks:
        try:
            return json.loads(json_blocks[0])
        except json.JSONDecodeError:
            pass

    # 2. 尝试从 ```...``` 代码块中提取
    code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
    for block in code_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # 3. 尝试直接解析整个响应
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # 4. 尝试正则提取 JSON 对象
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None

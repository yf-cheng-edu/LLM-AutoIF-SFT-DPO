import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

models = {
    "base": "/root/autodl-tmp/AutoIF-LLM/models/student/Qwen/Qwen2.5-1.5B",
}

test_cases = [
    ("How do I make my Wi-Fi secure?",
     "Construct the reply as if it's a telegram STOP"),
    ("Explain NLP briefly.",
     "Use only the first half of the alphabet (A-M)"),
    ("How to start a book club?",
     "Use words that end with '-ing'"),
]

def generate_robust(model, tokenizer, question, instruction):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Follow formatting instructions carefully."},
        {"role": "user",   "content": f"{question}\n\nFormatting requirement: {instruction}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    terminators = [tokenizer.eos_token_id, 151645, 151643]
    terminators = [t for t in terminators if t is not None]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    raw_response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    # 通用重复截断
    words = raw_response.split()
    for i in range(len(words) - 5):
        if len(set(words[i:i+5])) == 1:
            raw_response = ' '.join(words[:i])
            break
    
    # 已知乱码清理
    clean_response = re.split(r'\(egt\)|mPid|#aa|afone|isz', raw_response)[0].strip()
    
    if not clean_response:
        clean_response = "[模型未生成有效回答]"
    
    return clean_response

for model_name, model_path in models.items():
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print('='*60)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    for question, instruction in test_cases:
        response = generate_robust(model, tokenizer, question, instruction)
        print(f"\n📌 指令: {instruction}")
        print(f"💬 回答: {response}")
        print("-"*40)
    del model
    torch.cuda.empty_cache()
import requests

def generate_response(question, instruction):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Follow the user's formatting instructions strictly."},
            {"role": "user", "content": f"{question}\n\nConstraint: {instruction}"}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.1,  # 对应 repetition_penalty=1.15
        "stop": ["<|im_end|>", "<Task complete"]  # 对应原始 stop_signals
    }
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    
    return result["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    question = "How do I make my Wi-Fi secure?"
    instruction = "Construct the reply as if it's a telegram. Use 'STOP' at the end of each sentence."
    
    print("=" * 60)
    print(f"Question   : {question}")
    print(f"Instruction: {instruction}")
    print("=" * 60)
    
    response = generate_response(question, instruction)
    print("Response:")
    print(response)
    print("=" * 60)
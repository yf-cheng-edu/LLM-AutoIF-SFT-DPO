import sys, json, os
path = sys.argv[1] + "/config.json"
if not os.path.exists(path):
    print(f"找不到文件: {path}")
    sys.exit(0)

with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# 核心修复：强行注入 vLLM 需要的 factor 参数
if "rope_scaling" not in cfg or not isinstance(cfg["rope_scaling"], dict):
    cfg["rope_scaling"] = {"type": "default"}

cfg["rope_scaling"]["factor"] = 1.0

with open(path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)
print(f"✅ 成功给 {path} 注入补丁！")

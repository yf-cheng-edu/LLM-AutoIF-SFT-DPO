import os, json
for root, dirs, files in os.walk('models'):
    if 'config.json' in files:
        path = os.path.join(root, 'config.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if 'rope_scaling' in cfg:
                del cfg['rope_scaling']
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2)
                print(f'✅ 成功修复: {path}')
        except Exception as e:
            pass

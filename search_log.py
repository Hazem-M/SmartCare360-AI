import json

log_path = r'C:\Users\hazem\.gemini\antigravity\brain\dbcf95f6-8eea-48f3-b69f-a82ca1fc0401\.system_generated\logs\transcript_full.jsonl'

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            content = data.get('content', '')
            if 'transforms.Resize' in content and 'val_test_transforms' in content:
                print("====================================")
                idx = content.find('val_test_transforms')
                print(content[max(0, idx-200):idx+500])
        except Exception:
            pass

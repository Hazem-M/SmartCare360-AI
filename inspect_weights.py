import torch

checkpoint = torch.load(r'D:\Programing\College\Graduation Project\Final\Ai_2\chest_xray_results\best_overall_model_weights_only.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint.get('model_state', checkpoint)

for k, v in state_dict.items():
    if k.startswith('fc.'):
        print(f"{k}: {v.shape}")

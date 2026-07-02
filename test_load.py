import torch
import torchvision.models as models
import torch.nn as nn
import traceback

device = torch.device('cpu')
try:
    print("Trying load...")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 4))
    
    # Try with sequential dropout first, then fallback
    try:
        model.load_state_dict(torch.load(r'D:\Programing\College\Graduation Project\Final\Ai_2\chest_xray_results\best_overall_model_weights_only.pth', map_location=device, weights_only=True))
        print('Loaded OK with Dropout(0.2) + Linear')
    except:
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load(r'D:\Programing\College\Graduation Project\Final\Ai_2\chest_xray_results\best_overall_model_weights_only.pth', map_location=device, weights_only=True))
        print('Loaded OK with just Linear')
except Exception as e:
    print("Error:", e)
    traceback.print_exc()

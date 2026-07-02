import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import json
import numpy as np

def predict_image(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Model Info (Dynamic Architecture & Classes)
    model_info_path = r"d:\Ai\model_info.json"
    classes_path = r"d:\Ai\classes.txt"
    
    architecture = "densenet121" # Default fallback
    
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            info = json.load(f)
            class_names = info.get("classes", [])
            architecture = info.get("architecture", "densenet121")
    elif os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        print("model_info.json or classes file not found! Please run train.py first.")
        return

    # 2. بناء نفس النموذج الفائز وتحميل الأوزان المدربة
    model_path = r"d:\Ai\best_model.pth"
    if not os.path.exists(model_path):
        print("Model file not found! Please run train.py first.")
        return

    if architecture == "densenet121":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, len(class_names)))
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, len(class_names)))
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, len(class_names)))
    else:
        print(f"Unsupported architecture: {architecture}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. تجهيز الصورة (نفس الإعدادات المستخدمة في التدريب 512x512)
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # 4. Check if the image is likely an X-ray (advanced heuristic)
    def is_xray(img, tolerance=15):
        img_array = np.array(img)
        # Check 1: Is it Grayscale?
        # We use 95th percentile instead of mean to catch ANY colored objects (like a brown desk)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        diff_rg = np.abs(r.astype(int) - g.astype(int))
        diff_rb = np.abs(r.astype(int) - b.astype(int))
        diff_gb = np.abs(g.astype(int) - b.astype(int))
        
        p95_rg = np.percentile(diff_rg, 95)
        p95_rb = np.percentile(diff_rb, 95)
        p95_gb = np.percentile(diff_gb, 95)
        
        is_gray = max(p95_rg, p95_rb, p95_gb) <= tolerance
        
        if not is_gray:
            return False
            
        # Check 2: X-Ray Physical Properties (Background & Bones)
        img_gray = np.array(img.convert('L'))
        dark_pixels_ratio = np.sum(img_gray < 50) / img_gray.size
        bright_pixels_ratio = np.sum(img_gray > 200) / img_gray.size
        
        # Papers have mostly white background (bright_ratio > 40%)
        # Pure black images have no bones (bright_ratio < 0.1%)
        if dark_pixels_ratio < 0.05 or bright_pixels_ratio > 0.40 or bright_pixels_ratio < 0.001:
            return False
            
        return True

    if not is_xray(image):
        print("\n" + "=" * 40)
        print("  ❌ خطأ: هذه ليست صورة أشعة (X-Ray).")
        print("  يرجى إدخال مسار لصورة أشعة صحيحة للتحليل.")
        print("=" * 40 + "\n")
        return

    # 5. Transform and Predict
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 4. التوقع
    with torch.no_grad():
        outputs = model(input_batch)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    predicted_class = class_names[preds[0]]
    confidence = probabilities[preds[0]].item()

    # ترجمة النتيجة للعربية
    arabic_label = "مكسورة (Fractured)" if predicted_class == "Fractured" else "سليمة (Non-fractured)"

    print("\n" + "=" * 40)
    print(f"  Prediction : {predicted_class}")
    print(f"  التشخيص    : {arabic_label}")
    print(f"  Confidence : {confidence:.2f}%")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print('Example: python predict.py "d:\\Ai\\test_image.jpg"')
    else:
        predict_image(sys.argv[1])

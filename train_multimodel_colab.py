"""
==========================================================
  Multi-Model Bone Fracture Detection - Improved
==========================================================
  ميزات هذه النسخة المحدثة:
  - تقسيم البيانات: Train (70%), Val (15%), Test (15%)
  - استخدام WeightedRandomSampler فقط (بدون Double Weighting في الـ Loss)
  - تدريب كافة طبقات النموذج (Unfreeze) بدلاً من الطبقة الأخيرة فقط لزيادة الدقة
  - الاعتماد على F1-Score لاختيار أفضل نموذج بدلاً من Loss
  - إيجاد أفضل Threshold (عتبة) باستخدام بيانات الـ Validation
  - اختبار النموذج النهائي الفائز على بيانات الـ Test وتقييمه
==========================================================
"""

# =========================
# الخلية 1: توصيل Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report
from PIL import ImageFile

# إصلاح مشكلة الصور التالفة أو المقطوعة في الداتا ست
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# الخلية 2: الإعدادات (Config)
# =========================
DATA_DIR = "/content/drive/MyDrive/FracAtlas/images"
SAVE_DIR = "/content/drive/MyDrive/FracAtlas"

BATCH_SIZE = 16
NUM_EPOCHS_PER_MODEL = 15
PATIENCE = 5
IMG_SIZE = 512  # تم إعادته لـ 512 لرؤية الكسور الدقيقة، يمكن تغييره لـ 224 لتسريع التدريب
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# الخلية 3: محولات البيانات (Transforms)
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# الخلية 4: تجهيز البيانات وتقسيمها
# =========================
print("Loading dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)
dataset_size = len(full_dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# تطبيق الـ Augmentation على التدريب فقط
train_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

# Weighted Sampler لموازنة التدريب (لأن فئة السليم أكثر من الكسر بكثير)
train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_sample_count = np.array([train_targets.count(i) for i in range(num_classes)])
weight_per_class = 1. / class_sample_count
sample_weights = np.array([weight_per_class[t] for t in train_targets])

sample_weights = torch.from_numpy(sample_weights).double()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": train_size, "val": val_size}

print(f"Dataset split: Train: {train_size}, Val: {val_size}, Test: {test_size}")

# =========================
# الخلية 5: دالة بناء النموذج
# =========================
def build_model(model_name):
    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError("Unsupported model")

    # نترك الطبقات قابلة للتدريب (Unfreeze) لأن الصور الطبية تختلف عن ImageNet
    for param in model.parameters():
        param.requires_grad = True

    return model.to(device)

# =========================
# الخلية 6: دالة التدريب
# =========================
def train_model(model, model_name):
    print(f"\n🚀 Training {model_name}...")

    # نستخدم CrossEntropyLoss بدون أوزان إضافية (تم منع التعويض المزدوج)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS_PER_MODEL):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_PER_MODEL}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            # نستخدم macro لحساب المتوسط بغض النظر عن حجم الفئة
            epoch_f1 = f1_score(all_labels, all_preds, average="macro")

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)

                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print("  ✅ New best model!")
                else:
                    patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    return model, best_f1, best_loss

# =========================
# الخلية 7: تشغيل المقارنة واختيار الأفضل
# =========================
models_to_try = ["densenet121", "resnet50", "efficientnet_b0"]
results = {}

best_model = None
best_model_name = ""
best_f1_overall = 0

for m_name in models_to_try:
    model = build_model(m_name)
    trained_model, model_f1, model_loss = train_model(model, m_name)

    results[m_name] = {"f1": model_f1, "loss": model_loss}

    if model_f1 > best_f1_overall:
        best_f1_overall = model_f1
        best_model = copy.deepcopy(trained_model)
        best_model_name = m_name

print("\n🏆 Models Comparison:")
for m_name, metrics in results.items():
    print(f" - {m_name}: Val F1 = {metrics['f1']:.4f}, Val Loss = {metrics['loss']:.4f}")

print(f"\n🌟 The Best Model is: {best_model_name.upper()} with F1: {best_f1_overall:.4f} 🌟")

# =========================
# الخلية 8: ضبط العتبة (Threshold Tuning) على مجموعة الـ Validation
# =========================
print("\n🔧 Threshold Tuning for Best Model...")
best_model.eval()
val_probs, val_labels = [], []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Finding best threshold"):
        inputs = inputs.to(device)
        outputs = best_model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        val_probs.extend(probs.cpu().numpy())
        val_labels.extend(labels.numpy())

val_probs = np.array(val_probs)
val_labels = np.array(val_labels)

fractured_idx = class_names.index("Fractured")
fractured_probs = val_probs[:, fractured_idx]
fractured_labels = (val_labels == fractured_idx).astype(int)

best_threshold = 0.5
best_thresh_f1 = 0.0

for threshold in np.arange(0.20, 0.80, 0.02):
    preds_at_threshold = (fractured_probs >= threshold).astype(int)
    f1 = f1_score(fractured_labels, preds_at_threshold)
    if f1 > best_thresh_f1:
        best_thresh_f1 = f1
        best_threshold = threshold

print(f"✅ Optimal Threshold for 'Fractured': {best_threshold:.2f} (F1={best_thresh_f1:.4f})")

# =========================
# الخلية 9: حفظ النموذج والإعدادات
# =========================
torch.save(best_model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

with open(os.path.join(SAVE_DIR, "threshold.txt"), "w") as f:
    f.write(str(best_threshold))

model_info = {
    "architecture": best_model_name,
    "classes": class_names
}
with open(os.path.join(SAVE_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f)

print("💾 Saved best_model.pth, model_info.json, and threshold.txt")

# =========================
# الخلية 10: تقييم النموذج النهائي على مجموعة الـ Test المستقلة
# =========================
print("\n📊 Final Evaluation on Test Set...")
best_model.eval()
test_probs, test_labels = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device)
        outputs = best_model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(labels.numpy())

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)

# نطبق الـ Threshold الذي حصلنا عليه
test_fractured_probs = test_probs[:, fractured_idx]
final_preds = np.where(test_fractured_probs >= best_threshold, fractured_idx, 1 - fractured_idx)

print("\nClassification Report (Test Set):")
print(classification_report(test_labels, final_preds, target_names=class_names))

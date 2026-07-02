import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import numpy as np
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

load_dotenv()
# Initialize Gemini Keys (Rotation for avoiding limits)
API_KEYS = [
    os.getenv("GEMINI_API_KEY", "PUT_YOUR_API_KEY_HERE"),
    # أضف مفاتيحك الإضافية هنا لمضاعفة الليميت (كل مفتاح يعطيك 15 طلب إضافي في الدقيقة):
    # "YOUR_SECOND_API_KEY_HERE",
    # "YOUR_THIRD_API_KEY_HERE",
]
API_KEYS = [k for k in API_KEYS if k and k != "??_?????_??????_???"]

gemini_clients = [genai.Client(api_key=k) for k in API_KEYS]
has_gemini = len(gemini_clients) > 0
client_iterator = itertools.cycle(gemini_clients) if has_gemini else None

def get_gemini_client():
    return next(client_iterator) if has_gemini else None

def generate_fallback_report(disease_name: str, confidence: float, lang: str) -> dict:
    is_en = (lang == "en")
    is_normal = "Normal" in disease_name or "طبيعي" in disease_name or "سليم" in disease_name
    
    return {
        "disease_name": disease_name,
        "bone_name": disease_name,
        "fracture_type": ("None" if is_en else "لا يوجد") if is_normal else ("AI Assessment" if is_en else "تقييم ذكاء اصطناعي"),
        "severity": ("Normal" if is_en else "طبيعي") if is_normal else ("AI Assessment" if is_en else "تقييم ذكاء اصطناعي"),
        "report": (
            "Based on the AI model analysis, the medical image appears normal with no clear signs of abnormalities." if is_normal else f"The AI model detected '{disease_name}' with a confidence of {confidence:.2f}%. Please review the highlighted heatmap areas for visual indicators."
        ) if is_en else (
            "بناءً على تحليل نموذج الذكاء الاصطناعي، تبدو الصورة الطبية طبيعية ولا توجد علامات واضحة على وجود تشوهات." if is_normal else f"اكتشف نموذج الذكاء الاصطناعي وجود '{disease_name}' بنسبة ثقة {confidence:.2f}%. يرجى مراجعة الخريطة الحرارية المرفقة للمؤشرات البصرية."
        ),
        "treatment_plan": (
            "No specific medical intervention is required based on this image. However, if symptoms persist, please consult a healthcare professional. (Note: These are AI recommendations and not a substitute for a doctor)" if is_normal else "Please consult a specialized medical professional immediately for an accurate clinical diagnosis and treatment plan. (Note: These are AI recommendations and not a substitute for a doctor)"
        ) if is_en else (
            "لا حاجة لتدخل طبي بناءً على هذه الصورة. ولكن إذا استمرت الأعراض، يرجى استشارة طبيب مختص. (ملاحظة: هذه توصيات مبدئية من الذكاء الاصطناعي وليست بديلاً عن الطبيب)" if is_normal else "يرجى مراجعة طبيب مختص فوراً للحصول على تشخيص سريري دقيق وخطة علاج. (ملاحظة: هذه توصيات مبدئية من الذكاء الاصطناعي وليست بديلاً عن الطبيب)"
        )
    }

# 1. Initialize FastAPI app
app = FastAPI(title="Bone Fracture Detection API")

# Setup CORS to allow requests from any origin (e.g., your GitHub hosted site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the model, class names, and transforms
model = None
class_names = []
data_transforms = None
device = None

# Chest X-Ray Model
chest_model = None
chest_class_names = ["CARDIOMEGALY", "NORMAL", "PLEURAL_EFFUSION", "PNEUMONIA"]
chest_transforms = None

@app.on_event("startup")
def load_model():
    """ Load the model when the API starts up so it's ready to handle requests fast """
    global model, class_names, data_transforms, device
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Model Info (Dynamic Architecture & Classes)
    base_dir = os.path.dirname(__file__)
    model_info_path = os.path.join(base_dir, "model_info.json")
    classes_path = os.path.join(base_dir, "classes.txt")
    
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
        print("Warning: model_info.json or classes.txt not found. Cannot proceed.")
        return

    # Load Model dynamically
    model_path = os.path.join(base_dir, "best_model.pth")
    if os.path.exists(model_path):
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
    else:
        print("Warning: best_model.pth not found. Cannot proceed.")
        return

    # Load Threshold (If exists)
    threshold_path = os.path.join(base_dir, "threshold.txt")
    global optimal_threshold
    optimal_threshold = 0.5 # Default
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            optimal_threshold = float(f.read().strip())
        print(f"Loaded optimized threshold: {optimal_threshold}")
        
    # Image Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Model loaded successfully!")
    
    # Load Chest X-ray Model (ResNet50)
    print("Loading Chest X-ray Model...")
    chest_model_path = os.path.join(base_dir, "..", "Ai_2", "chest_xray_results", "best_overall_model_weights_only.pth")
    if os.path.exists(chest_model_path):
        try:
            global chest_model, chest_transforms
            chest_model = models.resnet50(weights=None)
            num_ftrs_chest = chest_model.fc.in_features
            # The original model might have had dropout=0.2 or 0.5, we will use 0.2 as standard, but often the final layer structure matters.
            # Assuming standard ResNet50 FC replacement:
            chest_model.fc = nn.Sequential(
                nn.Linear(num_ftrs_chest, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, len(chest_class_names))
            )
            checkpoint = torch.load(chest_model_path, map_location=device, weights_only=False)
            if 'model_state' in checkpoint:
                chest_model.load_state_dict(checkpoint['model_state'])
            else:
                chest_model.load_state_dict(checkpoint)
            chest_model = chest_model.to(device)
            chest_model.eval()
            
            # Use 224x224 for ResNet50
            chest_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("Chest X-ray Model loaded successfully!")
        except Exception as e:
            print(f"Error loading Chest X-ray Model: {e}")
    else:
        print(f"Warning: Chest model not found at {chest_model_path}")

@app.get("/")
async def get_index():
    # Serve the HTML frontend
    html_path = os.path.join(os.path.dirname(__file__), "test_website.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    return {"message": "Frontend not found, but API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form("ar")):
    if model is None:
        return {"error": "Model is not loaded."}
    
    try:
        # Read the image file uploaded by the user
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    # Validate image using Gemini
    if has_gemini and get_gemini_client():
        try:
            val_prompt = "Is this image a valid medical X-ray? Reply strictly with 'YES' or 'NO'."
            val_response = get_gemini_client().models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, val_prompt]
            )
            val_answer = val_response.text.strip().upper()
            if "YES" not in val_answer:
                error_msg = "Please upload a valid X-ray image." if lang == "en" else "عذراً، هذه الصورة لا تبدو كأشعة طبية صالحة للفحص. يرجى رفع صورة صحيحة."
                return {"success": False, "error": error_msg}
        except Exception as e:
            print("Gemini Image Validation Error:", e)
            # Fail securely: block prediction if Gemini validation throws an error (e.g. invalid key or network issue)
            error_msg = "Image validation service is currently unavailable. Please check your API key or network." if lang == "en" else f"حدث خطأ أثناء التحقق من الصورة باستخدام الذكاء الاصطناعي: {str(e)}"
            return {"success": False, "error": error_msg}

    # 2. Transform image and run through the model
    input_tensor = data_transforms(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probabilities_tensor = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get probability for "Fractured" (assuming it is class index 0 based on alphabetical order)
    fractured_idx = class_names.index("Fractured") if "Fractured" in class_names else 0
    fractured_prob = probabilities_tensor[fractured_idx].item()
    
    # Apply optimal threshold
    is_fractured = fractured_prob >= optimal_threshold
    if is_fractured:
        pred_idx = fractured_idx
        confidence = fractured_prob * 100
    else:
        pred_idx = 1 - fractured_idx
        confidence = (1 - fractured_prob) * 100
        
    predicted_class = class_names[pred_idx]
    if lang == "en":
        final_label = "Fractured" if predicted_class == "Fractured" else "Non-fractured"
    else:
        final_label = "مكسورة (Fractured)" if predicted_class == "Fractured" else "سليمة (Non-fractured)"

    # --- Generate Heatmap (Grad-CAM) if Fractured ---
    heatmap_base64 = None
    report_data = None
    
    if is_fractured:
        try:
            # 1. Grad-CAM Logic
            target_layers = []
            arch = model.__class__.__name__.lower()
            if "densenet" in arch:
                target_layers = [model.features[-1]]
            elif "resnet" in arch:
                target_layers = [model.layer4[-1]]
            elif "efficientnet" in arch:
                target_layers = [model.features[-1]]
                
            if target_layers:
                cam = GradCAM(model=model, target_layers=target_layers)
                targets = [ClassifierOutputTarget(fractured_idx)]
                
                # GradCAM expects requires_grad=True, but we're in inference. 
                # PyTorch Grad-CAM handles this internally, but we must run it outside torch.no_grad()
                with torch.enable_grad():
                    # Re-run forward pass with gradients enabled for CAM
                    grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0, :]
                
                img_scaled = np.array(image.resize((512, 512))) / 255.0
                visualization = show_cam_on_image(img_scaled, grayscale_cam, use_rgb=True)
                
                viz_image = Image.fromarray(visualization)
                buffered = io.BytesIO()
                viz_image.save(buffered, format="JPEG")
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print("Grad-CAM Error:", e)

        # 2. Medical Report Logic (Gemini API)
        if has_gemini and get_gemini_client():
            try:
                if lang == "en":
                    prompt = f"""
                    You are an expert AI Radiologist.
                    This image was analyzed by a diagnostic model, confirming a fracture with {confidence:.2f}% confidence.
                    Please look at this X-ray image and extract the following information in JSON format only (without any other text or markdown):
                    {{
                        "bone_name": "Name of the fractured bone (e.g., Forearm, Femur, Clavicle)",
                        "fracture_type": "Type of fracture (Simple, Compound, Spiral, Comminuted, etc.)",
                        "severity": "Severity (Mild, Moderate, Severe)",
                        "report": "Short medical report in English explaining the case as a doctor",
                        "treatment_plan": "Initial treatment advice (e.g., rest, splint, surgery) with a note: 'These are preliminary recommendations and not a substitute for a doctor'"
                    }}
                    """
                else:
                    prompt = f"""
                    أنت طبيب ذكاء اصطناعي خبير في الأشعة (Radiologist).
                    تم تحليل هذه الصورة بنموذج تشخيصي، وأكد وجود كسر بنسبة {confidence:.2f}%.
                    يرجى النظر إلى صورة الأشعة هذه واستخراج المعلومات التالية بصيغة JSON فقط (بدون أي نص آخر أو markdown):
                    {{
                        "bone_name": "اسم العظمة المكسورة (مثل الساعد، الفخذ، الترقوة)",
                        "fracture_type": "نوع الكسر (بسيط، مضاعف، حلزوني، متفتت، إلخ)",
                        "severity": "درجة الخطورة (بسيطة، متوسطة، شديدة)",
                        "report": "تقرير طبي قصير باللغة العربية يشرح الحالة كطبيب",
                        "treatment_plan": "نصيحة للعلاج المبدئي (مثل: راحة، جبيرة، تدخل جراحي) مع وضع ملاحظة: 'هذه توصيات مبدئية وليست بديلاً عن الطبيب'"
                    }}
                    """
                # We need to ensure the image is in a format Gemini accepts (PIL Image is fine)
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                )
                
                response = get_gemini_client().models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image],
                    config=config
                )
                
                # Extract JSON
                import re
                text = response.text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    report_data = json.loads(match.group(0))
                else:
                    report_data = json.loads(text)
            except Exception as e:
                print("Gemini API Error (Bone):", e)
                report_data = generate_fallback_report(final_label, confidence, lang)
        else:
            report_data = generate_fallback_report(final_label, confidence, lang)

    # Return the response as JSON (Key-Value)
    response_payload = {
        "success": True,
        "prediction": final_label,
        "confidence": round(confidence, 2)
    }
    
    if heatmap_base64:
        response_payload["heatmap"] = heatmap_base64
    if report_data:
        response_payload["report_data"] = report_data

    return response_payload

@app.post("/predict/chest-xray")
async def predict_chest_xray(file: UploadFile = File(...), lang: str = Form("ar")):
    if chest_model is None:
        return {"success": False, "error": "Chest X-ray model is not loaded."}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return {"success": False, "error": f"Invalid image file: {e}"}

    # Validate image using Gemini
    if has_gemini and get_gemini_client():
        try:
            val_prompt = "Is this image a valid medical Chest X-ray? Reply strictly with 'YES' or 'NO'."
            val_response = get_gemini_client().models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, val_prompt]
            )
            val_answer = val_response.text.strip().upper()
            if "YES" not in val_answer:
                error_msg = "Please upload a valid Chest X-ray image." if lang == "en" else "عذراً، هذه الصورة لا تبدو كأشعة طبية للصدر. يرجى رفع صورة صحيحة."
                return {"success": False, "error": error_msg}
        except Exception as e:
            print("Gemini Image Validation Error:", e)
            error_msg = "Image validation service is currently unavailable. Please check your API key or network." if lang == "en" else f"حدث خطأ أثناء التحقق من الصورة باستخدام الذكاء الاصطناعي: {str(e)}"
            return {"success": False, "error": error_msg}

    # Transform image and run through the model
    input_tensor = chest_transforms(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = chest_model(input_batch)
        probabilities_tensor = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get Pneumonia probability (Index 3 in chest_class_names)
    pneumonia_prob = probabilities_tensor[3].item() * 100
    
    if pneumonia_prob >= 50.0:
        final_label = "Pneumonia" if lang == "en" else "التهاب رئوي (Pneumonia)"
        confidence = pneumonia_prob
        predicted_class = "PNEUMONIA"
        pred_idx = 3
    else:
        final_label = "Normal" if lang == "en" else "طبيعي / لا يوجد التهاب رئوي"
        confidence = 100.0 - pneumonia_prob
        predicted_class = "NORMAL"

    # --- Generate Heatmap (Grad-CAM) ---
    heatmap_base64 = None
    report_data = None
    
    if predicted_class != "NORMAL":
        try:
            # 1. Grad-CAM Logic
            target_layers = [chest_model.layer4[-1]]
            cam = GradCAM(model=chest_model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(pred_idx)]
            
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0, :]
            
            img_scaled = np.array(image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(img_scaled, grayscale_cam, use_rgb=True)
            
            viz_image = Image.fromarray(visualization)
            buffered = io.BytesIO()
            viz_image.save(buffered, format="JPEG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print("Grad-CAM Error:", e)

        # 2. Medical Report Logic (Gemini API)
        if has_gemini and get_gemini_client():
            try:
                if lang == "en":
                    prompt = f"""
                    You are an expert AI Radiologist.
                    This chest X-ray image was analyzed by a diagnostic model, confirming "{final_label}" with {confidence:.2f}% confidence.
                    Please look at this X-ray image and extract the following information in JSON format only (without any other text or markdown):
                    {{
                        "disease_name": "{final_label}",
                        "severity": "Severity (Mild, Moderate, Severe) based on what you see and know about the disease",
                        "report": "Short medical report in English explaining the case as a radiologist",
                        "treatment_plan": "Initial treatment advice with a note: 'These are preliminary recommendations and not a substitute for a doctor'"
                    }}
                    """
                else:
                    prompt = f"""
                    أنت طبيب ذكاء اصطناعي خبير في الأشعة (Radiologist).
                    تم تحليل صورة أشعة الصدر هذه بنموذج تشخيصي، وأكد وجود "{final_label}" بنسبة {confidence:.2f}%.
                    يرجى النظر إلى صورة الأشعة هذه واستخراج المعلومات التالية بصيغة JSON فقط (بدون أي نص آخر أو markdown):
                    {{
                        "disease_name": "{final_label}",
                        "severity": "درجة الخطورة (بسيطة، متوسطة، شديدة) بناءً على ما تراه وتعرفه عن المرض",
                        "report": "تقرير طبي قصير باللغة العربية يشرح الحالة كطبيب أشعة",
                        "treatment_plan": "نصيحة للعلاج المبدئي مع وضع ملاحظة: 'هذه توصيات مبدئية وليست بديلاً عن الطبيب'"
                    }}
                    """
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    ]
                )
                response = get_gemini_client().models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image],
                    config=config
                )
                import re
                text = response.text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    report_data = json.loads(match.group(0))
                else:
                    report_data = json.loads(text)
            except Exception as e:
                print("Gemini API Error (Chest):", e)
                report_data = generate_fallback_report(final_label, confidence, lang)
        else:
            report_data = generate_fallback_report(final_label, confidence, lang)

    response_payload = {
        "success": True,
        "prediction": final_label,
        "confidence": round(confidence, 2),
        "is_normal": predicted_class == "NORMAL"
    }
    if heatmap_base64:
        response_payload["heatmap"] = heatmap_base64
    if report_data:
        response_payload["report_data"] = report_data

    return response_payload

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    user_role: str = "guest"
    history: List[ChatMessage] = []
    platform_data: str = ""
    lang: str = "ar"

class SummarizeRequest(BaseModel):
    conversation: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not has_gemini or not get_gemini_client():
        reply = "عذراً، خدمة الذكاء الاصطناعي غير متاحة حالياً." if request.lang == "ar" else "Sorry, the AI service is currently unavailable."
        return {"success": False, "reply": reply}
    
    try:
        if request.lang == "en":
            user_type = "Doctor" if request.user_role == "doctor" else "Patient" if request.user_role == "patient" else "Visitor"
            system_instruction = f"""
            You are the AI technical support assistant for "Smart Care 360".
            The person talking to you is a: {user_type}.
            Platform data: {request.platform_data}
            Services we provide:
            1. Bone fracture detection using X-ray.
            2. Heart disease prediction.
            3. Brain tumor detection.
            4. Chat with doctors and book appointments.
            
            Your tasks:
            - Help the user navigate the website.
            - Answer questions about available doctors based on the platform data.
            - Use Markdown for bold text, links, and lists.
            - Do not provide any medical diagnoses. Your job is technical support and guiding the user.
            - Speak in a friendly and simple tone. Always reply in English.
            """
        else:
            user_type = "طبيب" if request.user_role == "doctor" else "مريض" if request.user_role == "patient" else "زائر"
            system_instruction = f"""
            أنت مساعد الدعم الفني لموقع "Smart Care 360". 
            الشخص الذي يتحدث معك هو: {user_type}.
            معلومات إضافية عن المنصة حالياً: {request.platform_data}
            الخدمات التي يقدمها الموقع:
            1. تشخيص كسور العظام عبر الأشعة السينية بالذكاء الاصطناعي.
            2. تشخيص أمراض القلب بالذكاء الاصطناعي.
            3. تشخيص أورام المخ بالذكاء الاصطناعي.
            4. محادثات فورية بين الأطباء والمرضى، وحجز المواعيد.
            
            مهمتك:
            - مساعدة المستخدم في تصفح الموقع والتعامل معه.
            - استخدم المعلومات الإضافية للإجابة عن الأطباء المتاحين إذا سأل المستخدم عن ذلك.
            - استخدم تنسيق Markdown لتمييز الكلمات الهامة (Bold)، والروابط، والقوائم.
            - لا تقدم أي تشخيص طبي إطلاقاً. مهمتك هي المساعدة التقنية وتوجيه المريض لرفع الأشعة أو التحدث لطبيب.
            - تحدث بلغة ودية ومبسطة، وإذا سألك بلغة أجنبية أجب بنفس اللغة.
            """
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        
        contents = []
        for msg in request.history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.text)]))
            
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=request.message)]))
        
        response = get_gemini_client().models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=config
        )
        return {"success": True, "reply": response.text}
    except Exception as e:
        error_str = str(e).lower()
        if "503" in error_str or "unavailable" in error_str or "high demand" in error_str:
            reply_msg = "عذراً، المساعد الذكي يواجه ضغطاً كبيراً في الوقت الحالي. يرجى المحاولة مرة أخرى بعد قليل." if request.lang == "ar" else "Sorry, the AI assistant is currently facing high demand. Please try again later."
        else:
            reply_msg = f"عذراً، حدث خطأ مؤقت أثناء الاتصال. يرجى المحاولة لاحقاً. ({str(e)})" if request.lang == "ar" else f"Sorry, a temporary connection error occurred. Please try again later. ({str(e)})"
        return {"success": False, "reply": reply_msg}

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    if not has_gemini or not get_gemini_client():
        return {"success": False, "reply": "خدمة الذكاء الاصطناعي غير متاحة."}
    
    try:
        prompt = f"""
        أنت مساعد طبي ذكي. قم بقراءة هذه المحادثة بين الطبيب والمريض وتلخيص حالة المريض، الأعراض، والنقاط الهامة في شكل نقاط واضحة وموجزة لتسهيل قراءتها على الطبيب باستخدام Markdown.
        المحادثة:
        {request.conversation}
        """
        
        config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )

        response = get_gemini_client().models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=config
        )
        return {"success": True, "reply": response.text}
    except Exception as e:
        import traceback
        err_msg = str(e)
        return {"success": False, "reply": f"حدث خطأ أثناء التلخيص: {err_msg}"}

if __name__ == "__main__":
    import uvicorn
    import os
    # Bind to 0.0.0.0 to support deployment environments like Render, Heroku, etc.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

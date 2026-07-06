from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import itertools
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize Gemini Keys (Rotation for avoiding limits)
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
    os.getenv("GEMINI_API_KEY_5", ""),
    os.getenv("GEMINI_API_KEY_6", ""),
    os.getenv("GEMINI_API_KEY_7", ""),
    os.getenv("GEMINI_API_KEY_8", "")
]
API_KEYS = [k for k in API_KEYS if k and k != "??_?????_??????_???"]

gemini_clients = [genai.Client(api_key=k) for k in API_KEYS]
has_gemini = len(gemini_clients) > 0
client_iterator = itertools.cycle(gemini_clients) if has_gemini else None

def get_gemini_client():
    return next(client_iterator) if has_gemini else None

app = FastAPI(title="Smart Care 360 AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_index():
    html_path = os.path.join(os.path.dirname(__file__), "test_website.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    return {"message": "Chatbot API is running efficiently."}

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
        
        max_retries = len(gemini_clients)
        for attempt in range(max_retries):
            try:
                response = get_gemini_client().models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents,
                    config=config
                )
                return {"success": True, "reply": response.text}
            except Exception as e:
                error_str = str(e).lower()
                # If quota exhausted, try next key
                if "429" in error_str or "exhausted" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        continue
                
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

        max_retries = len(gemini_clients)
        for attempt in range(max_retries):
            try:
                response = get_gemini_client().models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt],
                    config=config
                )
                return {"success": True, "reply": response.text}
            except Exception as e:
                error_str = str(e).lower()
                # If quota exhausted, try next key
                if "429" in error_str or "exhausted" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        continue
                
                return {"success": False, "reply": f"حدث خطأ أثناء التلخيص: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 to support deployment environments like Render, Heroku, etc.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

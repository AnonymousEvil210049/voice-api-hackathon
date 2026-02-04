import sys
import subprocess
import os
import importlib
from dotenv import load_dotenv 

# Load the vault (.env file)
load_dotenv()

# ==========================================
# 0. SELF-HEALING INSTALLER
# ==========================================
def install_requirements():
    requirements = {
        "uvicorn": "uvicorn",
        "fastapi": "fastapi",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "numpy": "numpy",
        "requests": "requests",
        "pydantic": "pydantic",
        "torch": "torch",
        "transformers": "transformers",
        "scipy": "scipy",
        "sounddevice": "sounddevice",
        "python-dotenv": "dotenv"
    }
    
    missing = []
    for lib, package in requirements.items():
        try:
            importlib.import_module(lib)
        except ImportError:
            missing.append(package)
            
    if missing:
        print(f"⚠️  Missing libraries: {', '.join(missing)}")
        print("⏳ Auto-installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["python-multipart"])
            print("✅ Installed! Starting app...")
        except Exception as e:
            print(f"❌ Install failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    install_requirements()

# ==========================================
# 1. IMPORTS & CONFIG
# ==========================================
import uvicorn
import base64
import io
import librosa
import numpy as np
import soundfile as sf
import requests
import tempfile
import warnings
import sounddevice as sd 
import scipy.io.wavfile as wav 
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# SECURE RETRIEVAL (No Hardcoding)
# ---------------------------------------------------------
SECRET_API_KEY = os.getenv("API_KEY")

if not SECRET_API_KEY:
    # Fallback only if the user forgot to set the env var
    print("❌ ERROR: API_KEY is missing from environment variables!")
# ---------------------------------------------------------

# ==========================================
# 2. AI MODEL SETUP
# ==========================================
print("⏳ Initializing System...")
AI_PIPELINE = None
USE_AI_MODEL = False

try:
    from transformers import pipeline
    MODEL_NAME = "dima806/deepfake_audio_detection" 
    print(f"🚀 Loading AI Model: {MODEL_NAME}...")
    AI_PIPELINE = pipeline("audio-classification", model=MODEL_NAME)
    USE_AI_MODEL = True
    print("✅ AI Model Loaded!")
except Exception as e:
    print(f"⚠️  AI Load Failed: {e}. Switching to Heuristic Mode.")
    USE_AI_MODEL = False

# ==========================================
# 3. API SETUP
# ==========================================
app = FastAPI(title="Agentic Honey-Pot Voice Detector")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str 
    audioBase64: str

def analyze_heuristic(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr_var = np.var(librosa.feature.zero_crossing_rate(y))
    
    if flatness < 0.005: 
        return "AI_GENERATED", 0.92, "Heuristic: Low spectral flatness."
    elif zcr_var < 0.0002:
        return "AI_GENERATED", 0.85, "Heuristic: Low ZCR variance."
    return "HUMAN", 0.89, "Heuristic: Natural frequency variance."

def analyze_voice(y, sr):
    if not USE_AI_MODEL or AI_PIPELINE is None:
        return analyze_heuristic(y, sr)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            temp_path = t.name
        
        sf.write(temp_path, y, sr)
        preds = AI_PIPELINE(temp_path)
        
        top = preds[0]
        label = top['label'].lower()
        score = top['score']
        
        try: os.remove(temp_path)
        except: pass

        if "fake" in label or "generated" in label:
            return "AI_GENERATED", round(score, 2), "AI Model detected synthetic patterns."
        return "HUMAN", round(score, 2), "AI Model verified natural characteristics."

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return analyze_heuristic(y, sr)

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(...)):
    # Verify the key matches the one in the vault
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(403, "Invalid Key")
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_io = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_io, sr=16000)
        cls, score, reason = analyze_voice(y, sr)
        return {
            "status": "success",
            "language": request.language,
            "classification": cls,
            "confidenceScore": score,
            "explanation": reason
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ==========================================
# 4. TESTER
# ==========================================
def record_audio(duration=5, fs=44100):
    print(f"🔴 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording, fs

def run_tester():
    print("\n" + "="*40)
    print("   🎤  VOICE DETECTION SUPER-TESTER  🎤")
    print("="*40)
    
    if not SECRET_API_KEY:
        print("❌ Cannot test: API_KEY not found in environment.")
        return

    print("Target Server:")
    print("1. Localhost (Default)")
    print("2. Custom URL (e.g. Render)")
    srv = input("Select [1/2]: ").strip()
    
    url = "http://127.0.0.1:8000/api/voice-detection"
    if srv == "2":
        url = input("Enter full URL: ").strip()

    while True:
        print("\nOptions: [1] File  [2] Live Mic  [3] Exit")
        choice = input("Select: ").strip()
        if choice == "3": break
        
        b64_data = ""
        if choice == "1":
            path = input("Enter file path: ").strip().strip('"')
            if not os.path.exists(path) or os.path.isdir(path): continue
            with open(path, "rb") as f: b64_data = base64.b64encode(f.read()).decode('utf-8')
        elif choice == "2":
            try:
                my_rec, fs = record_audio()
                temp_wav = "temp_mic.wav"
                wav.write(temp_wav, fs, my_rec)
                with open(temp_wav, "rb") as f: b64_data = base64.b64encode(f.read()).decode('utf-8')
                os.remove(temp_wav)
            except Exception as e: print(e); continue
        
        if b64_data:
            print(f"🚀 Sending to {url}...")
            try:
                # Use the secure key from the environment
                res = requests.post(url, 
                    json={"language": "English", "audioFormat": "mp3", "audioBase64": b64_data},
                    headers={"x-api-key": SECRET_API_KEY}
                )
                if res.status_code == 200:
                    data = res.json()
                    print(f"\n📝 {data['classification']} ({data['confidenceScore']})")
                else: print(f"❌ Error: {res.text}")
            except Exception as e: print(f"❌ Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tester()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
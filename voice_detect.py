import sys
import os
import base64
import io
import tempfile
import logging
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
# YOUR RENDER URL (Update this after deployment if it changes)
RENDER_URL = "https://voice-api-hackathon.onrender.com/api/voice-detection"
# YOUR API KEY
SECRET_API_KEY = "sk_prod_9a8b4c3d2e1f5a6b7c8d9e0f1a2b3c4d"

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. SERVER LOGIC (Runs on Render)
# ==========================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

# Server Imports
try:
    import uvicorn
    import librosa
    import numpy as np
    import soundfile as sf
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    pass 

# AI Model Setup (The "Honey Pot")
AI_PIPELINE = None
USE_AI_MODEL = False
try:
    from transformers import pipeline
    # Load AI, but handle failure gracefully
    AI_PIPELINE = pipeline("audio-classification", model="dima806/deepfake_audio_detection")
    USE_AI_MODEL = True
except Exception as e:
    logger.warning(f"⚠️ AI Model skipped: {e}")

app = FastAPI(title="Agentic Voice Detector")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str 
    audioBase64: str

def analyze_heuristic(y, sr):
    """Fallback Math Mode: Impossible to crash."""
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr_var = np.var(librosa.feature.zero_crossing_rate(y))
    
    if flatness < 0.005: 
        return "AI_GENERATED", 0.92, "Heuristic: Low spectral flatness."
    elif zcr_var < 0.0002:
        return "AI_GENERATED", 0.85, "Heuristic: Low signal variance."
    
    return "HUMAN", 0.89, "Heuristic: Natural variance."

def analyze_voice(y, sr):
    """Decides: Use AI? If fails, use Math."""
    if USE_AI_MODEL and AI_PIPELINE:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
                temp_path = t.name
            sf.write(temp_path, y, sr)
            preds = AI_PIPELINE(temp_path)
            try: os.remove(temp_path)
            except: pass
            
            top = preds[0]
            label = top['label'].lower()
            if "fake" in label or "generated" in label:
                return "AI_GENERATED", round(top['score'], 2), "AI Detected Synthetic Pattern."
            return "HUMAN", round(top['score'], 2), "AI Verified Human Pattern."
        except:
            return analyze_heuristic(y, sr)
            
    return analyze_heuristic(y, sr)

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
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
# 3. CLIENT LOGIC (Runs on Laptop)
# ==========================================
def run_client():
    print("🎙️  CLIENT MODE ACTIVATED")
    try:
        import sounddevice as sd
        import requests
        import scipy.io.wavfile as wav
    except ImportError:
        print("❌ INSTALL LIBS: pip install sounddevice requests scipy")
        return

    duration = 5
    fs = 16000
    print(f"🔴 Recording {duration}s... SPEAK NOW!")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    virtual_file = io.BytesIO()
    wav.write(virtual_file, fs, rec)
    virtual_file.seek(0)
    b64_audio = base64.b64encode(virtual_file.read()).decode('utf-8')

    payload = {"language": "English", "audioFormat": "mp3", "audioBase64": b64_audio}
    headers = {"x-api-key": SECRET_API_KEY}

    print(f"🚀 Sending to: {RENDER_URL}")
    try:
        resp = requests.post(RENDER_URL, json=payload, headers=headers)
        print(resp.json())
    except Exception as e:
        print(f"❌ Error: {e}")

# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # If ran with --client, acts as microphone. Otherwise acts as server.
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        run_client()
    else:
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
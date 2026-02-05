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
# REPLACE THIS WITH YOUR RENDER URL (Must end in /api/voice-detection)
RENDER_URL = "https://voice-api-hackathon.onrender.com/api/voice-detection"

# YOUR OFFICIAL API KEY
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

# Server Imports (These run on the cloud)
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
# Tries to load the brain. If it's too heavy for the free server, it skips to Math Mode.
AI_PIPELINE = None
USE_AI_MODEL = False
try:
    from transformers import pipeline
    # We use a distinct deepfake detection model
    AI_PIPELINE = pipeline("audio-classification", model="dima806/deepfake_audio_detection")
    USE_AI_MODEL = True
    logger.info("✅ AI Neural Network Loaded Successfully.")
except Exception as e:
    logger.warning(f"⚠️ AI Model skipped (System is in Light Mode): {e}")

app = FastAPI(title="Agentic Voice Detector")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str 
    audioBase64: str

def analyze_heuristic(y, sr):
    """
    FALLBACK MODE: Uses signal processing (Math) if AI fails.
    This guarantees you ALWAYS get a result and never crash.
    """
    # 1. Spectral Flatness: AI voices are often 'too perfect' (flat)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # 2. Zero Crossing Rate: Measures robotic consistency
    zcr_var = np.var(librosa.feature.zero_crossing_rate(y))
    
    if flatness < 0.005: 
        return "AI_GENERATED", 0.92, "Heuristic: Abnormally low spectral flatness detected."
    elif zcr_var < 0.0002:
        return "AI_GENERATED", 0.85, "Heuristic: Signal variance is too consistent (robotic)."
    
    return "HUMAN", 0.89, "Heuristic: Natural frequency variance observed."

def analyze_voice(y, sr):
    """
    HYBRID BRAIN: 
    1. Tries to use the AI Model.
    2. If AI crashes or is unavailable, switches to Math (Heuristic).
    """
    if USE_AI_MODEL and AI_PIPELINE:
        try:
            # Transformers need a file path, so we create a temp one
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
                temp_path = t.name
            sf.write(temp_path, y, sr)
            
            preds = AI_PIPELINE(temp_path)
            
            # Cleanup temp file
            try: os.remove(temp_path)
            except: pass
            
            top = preds[0]
            label = top['label'].lower()
            score = top['score']

            if "fake" in label or "generated" in label:
                return "AI_GENERATED", round(score, 2), "AI Neural Network detected synthetic patterns."
            return "HUMAN", round(score, 2), "AI Neural Network verified natural human characteristics."
        except Exception as e:
            logger.error(f"AI Engine Glitch: {e}. Switching to Agentic Fallback.")
            return analyze_heuristic(y, sr)
            
    return analyze_heuristic(y, sr)

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # Security Check
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    try:
        # Decode the Base64 audio from the judge
        audio_bytes = base64.b64decode(request.audioBase64)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Run Analysis
        cls, score, reason = analyze_voice(y, sr)
        
        # Return strict JSON format
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
    """
    This function ONLY runs when you explicitly start Client Mode.
    It records your voice and sends it to the server.
    """
    print("\n" + "="*40)
    print("🎙️  AGENTIC CLIENT MODE ACTIVATED 🎙️")
    print("="*40)
    
    # Lazy imports so the server doesn't crash on Render
    try:
        import sounddevice as sd
        import requests
        import scipy.io.wavfile as wav
    except ImportError:
        print("❌ MISSING LIBRARIES! On your laptop, run:")
        print("pip install sounddevice requests scipy")
        return

    duration = 5
    fs = 16000
    print(f"🔴 Recording for {duration} seconds... SPEAK NOW!")
    
    # Record Audio
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording Complete. Packaging...")

    # Convert to Base64
    virtual_file = io.BytesIO()
    wav.write(virtual_file, fs, rec)
    virtual_file.seek(0)
    b64_audio = base64.b64encode(virtual_file.read()).decode('utf-8')

    # Send to Cloud
    payload = {
        "language": "English",
        "audioFormat": "mp3", 
        "audioBase64": b64_audio
    }
    headers = {"x-api-key": SECRET_API_KEY}

    print(f"🚀 Sending to Cloud Brain: {RENDER_URL}")
    
    try:
        start_time = time.time()
        resp = requests.post(RENDER_URL, json=payload, headers=headers)
        duration = time.time() - start_time
        
        print("\n" + "⬇️  SERVER RESPONSE  ⬇️")
        print(f"⏱️ Time Taken: {round(duration, 2)}s")
        print(resp.json())
        print("="*40 + "\n")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # If specific flag is passed, run Client. Otherwise, run Server.
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        run_client()
    else:
        # Default to Server Mode (Render runs this automatically)
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
import requests
import base64
import os

# Configuration
API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_hackathon_winner_2026"  # Must match the one in main.py

def file_to_base64(file_path):
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_string

def test_api():
    print("--- Voice Detection API Tester ---")
    
    # 1. Get file path from user
    file_path = input("Enter the path to an MP3 file (e.g., C:\\music\\test.mp3): ").strip()
    
    # Remove quotes if the user copied them as path
    if file_path.startswith('"') and file_path.endswith('"'):
        file_path = file_path[1:-1]

    if not os.path.exists(file_path):
        print("❌ Error: File not found!")
        return

    print("Converting audio to Base64...")
    try:
        b64_audio = file_to_base64(file_path)
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return

    # 2. Prepare the payload (Must match schemas.py)
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": b64_audio
    }

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # 3. Send Request
    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print("\n--- API RESPONSE ---")
        print(f"Status Code: {response.status_code}")
        print("Body:", response.json())
        
        if response.status_code == 200:
            print("\n✅ SUCCESS! Your API is working perfectly.")
        else:
            print("\n⚠️ ISSUE DETECTED. Check the error message above.")

    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR: Is 'main.py' running? Make sure to run 'python main.py' in a separate terminal.")

if __name__ == "__main__":
    test_api()
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import torch
import tempfile
import shutil

app = FastAPI()

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = pipeline("automatic-speech-recognition", model="therealbee/whisper-small-ha-bible-tts", device=0 if device == "cuda" else -1)

@app.get("/")
async def root():
    return {"message": "Welcome to the Hausa Speech-to-Text API!"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            shutil.copyfileobj(file.file, temp_audio)
            temp_audio_path = temp_audio.name
        result = whisper_model(temp_audio_path)
        return {"transcription": result["text"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get Render-assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

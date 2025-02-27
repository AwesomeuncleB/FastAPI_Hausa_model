from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import torch
import tempfile
import shutil

# Initialize FastAPI app
app = FastAPI()

# Root route (fix for 404 error)
@app.get("/")
async def root():
    return {"message": "Welcome to the Hausa Speech-to-Text API!"}

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = pipeline("automatic-speech-recognition", model="therealbee/whisper-small-ha-bible-tts", device=0 if device == "cuda" else -1)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            shutil.copyfileobj(file.file, temp_audio)
            temp_audio_path = temp_audio.name

        # Perform transcription
        result = whisper_model(temp_audio_path)

        return {"transcription": result["text"]}

    except Exception as e:
        return {"error": str(e)}

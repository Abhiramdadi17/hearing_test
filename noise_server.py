from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import noisereduce as nr
import scipy.io.wavfile as wav
import numpy as np
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/denoise-audio/")
async def denoise_audio(file: UploadFile = File(...)):
    # Save uploaded file
    raw_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    # Read and reduce noise
    rate, data = wav.read(raw_path)

    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)  # Convert to mono

    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # Save processed file
    cleaned_path = os.path.join(PROCESSED_DIR, f"cleaned_{os.path.basename(raw_path)}")
    wav.write(cleaned_path, rate, reduced_noise.astype(np.int16))

    return FileResponse(cleaned_path, media_type="audio/wav", filename="denoised_audio.wav")

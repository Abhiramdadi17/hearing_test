from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import noisereduce as nr
import numpy as np
import soundfile as sf
import os

app = FastAPI()

# Allow Flutter app to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Noise Reduction API is running"}

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"cleaned_{file.filename}"

    try:
        # Save uploaded audio to disk
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Read audio
        audio, rate = sf.read(input_path, dtype='float32')

        # Ensure mono for smaller memory usage
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        chunk_size = rate * 2  # 2 seconds per chunk
        cleaned_audio = []

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start+chunk_size]
            reduced_chunk = nr.reduce_noise(y=chunk, sr=rate)
            cleaned_audio.append(reduced_chunk)

        cleaned_audio = np.concatenate(cleaned_audio)

        # Save cleaned audio
        sf.write(output_path, cleaned_audio, rate)

        # Return file as hex string
        with open(output_path, "rb") as f:
            content = f.read()

        return {
            "filename": output_path,
            "data": content.hex()
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

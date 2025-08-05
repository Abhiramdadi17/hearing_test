from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import noisereduce as nr
import numpy as np
import soundfile as sf
import os

app = FastAPI()

# Optional: Enable CORS so your Flutter app or web client can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now; restrict this in production
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
        # Save uploaded audio
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Read audio
        audio, rate = sf.read(input_path)

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=audio, sr=rate)

        # Save cleaned audio
        sf.write(output_path, reduced_noise, rate)

        # Return the cleaned file as binary content
        with open(output_path, "rb") as f:
            content = f.read()

        return {
            "filename": output_path,
            "data": content.hex()  # You'll decode this on client side
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up temp files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

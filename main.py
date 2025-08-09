from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import noisereduce as nr
import numpy as np
import soundfile as sf
import os

app = FastAPI()

# Enable CORS
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
        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Read audio
        audio, rate = sf.read(input_path)

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=audio, sr=rate)

        # Save cleaned audio
        sf.write(output_path, reduced_noise, rate)

        with open(output_path, "rb") as f:
            content = f.read()

        return {
            "filename": output_path,
            "data": content.hex()
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use Railway's PORT or default to 10000
    uvicorn.run(app, host="0.0.0.0", port=port)

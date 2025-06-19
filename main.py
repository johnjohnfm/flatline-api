from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
import numpy as np
import os
import uuid
import librosa
import soundfile as sf
import io
import tempfile

app = FastAPI()

# CORS configuration (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "FLATLINE API is active."}

@app.post("/neutralize/")
async def neutralize_audio(file: UploadFile = File(...)):
    try:
        # Read file into a temporary buffer
        contents = await file.read()
        buffer = io.BytesIO(contents)

        # Load audio with librosa for compatibility with float32
        y, sr = librosa.load(buffer, sr=None, mono=True)  # Auto-detect sample rate

        # Convert to dB
        S = np.abs(librosa.stft(y))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # Calculate average frequency response (mean curve)
        mean_curve = S_db.mean(axis=1, keepdims=True)

        # Invert the curve to flatten it
        S_db_flat = S_db - mean_curve

        # Convert back to amplitude
        S_flat = librosa.db_to_amplitude(S_db_flat)

        # Reconstruct time-domain signal
        y_flat = librosa.istft(S_flat)

        # Write to temp file as 32-bit float WAV
        out_path = os.path.join(tempfile.gettempdir(), f"flatline-{uuid.uuid4().hex}.wav")
        sf.write(out_path, y_flat, sr, subtype='FLOAT')

        return FileResponse(out_path, media_type="audio/wav", filename="flatlined.wav")

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

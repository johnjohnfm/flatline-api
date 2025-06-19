from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io

app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Load audio into NumPy array
        data, samplerate = sf.read(file.file, dtype='float32')  # float32 guarantees precision, lower RAM than float64

        # STEP 1: Analyze spectral content (FFT-based)
        spectrum = np.abs(np.fft.rfft(data, axis=0))
        avg_spectrum = np.mean(spectrum, axis=0)
        neutral_curve = 1.0 / (avg_spectrum + 1e-8)  # avoid division by zero

        # STEP 2: Normalize spectrum
        freq_bins = np.fft.rfftfreq(data.shape[0], d=1/samplerate)
        data_fft = np.fft.rfft(data, axis=0)
        data_fft *= neutral_curve[:, np.newaxis] if data_fft.ndim > 1 else neutral_curve
        processed = np.fft.irfft(data_fft, axis=0)

        # Ensure the audio doesn't clip
        processed = np.clip(processed, -1.0, 1.0)

        # STEP 3: Encode to .wav
        buffer = io.BytesIO()
        sf.write(buffer, processed, samplerate, format='WAV', subtype='FLOAT')
        buffer.seek(0)

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=neutralized.wav"}
        )

    except Exception as e:
        return {"error": "Processing Failed", "details": str(e)}

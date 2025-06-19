from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
import librosa
import soundfile as sf
import io
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# /analyze/: Spectral Clustering
# -----------------------------
class Config(BaseModel):
    k: int
    similarity: Literal["gaussian", "knn", "epsilon"]
    laplacian_type: Literal["unnormalized", "normalized", "random_walk"]

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), config: Config = None):
    df = pd.read_csv(io.StringIO((await file.read()).decode()))
    X = df.values

    if config.similarity == "gaussian":
        sim = rbf_kernel(X)
    elif config.similarity == "knn":
        sim = kneighbors_graph(X, n_neighbors=10, mode='connectivity').toarray()
    elif config.similarity == "epsilon":
        D = pairwise_distances(X)
        sim = (D < 0.5).astype(float)

    norm = config.laplacian_type != "unnormalized"
    lap, _ = csgraph_laplacian(sim, normed=norm, return_diag=True)

    eigvals, eigvecs = eigsh(lap, k=config.k + 1, which='SM')
    eigvecs = eigvecs[:, 1:config.k+1]

    labels = KMeans(n_clusters=config.k).fit_predict(eigvecs)

    return {
        "clusters": labels.tolist(),
        "eigenvalues": eigvals.tolist()
    }

# -----------------------------
# /neutralize/: Audio Flattening
# -----------------------------
@app.post("/neutralize/")
async def neutralize(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")

    if not file.filename.endswith(".wav"):
        return JSONResponse({"error": "Only .wav files supported"}, status_code=400)

    try:
        # Read and buffer the file
        data = await file.read()
        buffer = io.BytesIO(data)
        buffer.seek(0)

        # Load with soundfile (preserves 32-bit float and stereo)
        y, sr = sf.read(buffer, dtype='float32')
        logging.info(f"Sample rate: {sr}, shape: {y.shape}")

        # Optional: convert to mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)
            logging.info("Converted to mono.")

        # Compute STFT
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        mag, phase = np.abs(S), np.angle(S)
        logging.info("STFT complete.")

        # Compute average spectrum
        avg_spectrum = mag.mean(axis=1)
        avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum)

        # Create neutralizing curve
        neutral_curve_db = -avg_spectrum_db + np.max(avg_spectrum_db)
        neutral_curve = librosa.db_to_amplitude(neutral_curve_db)

        # Apply neutralizing curve
        for i in range(mag.shape[1]):
            mag[:, i] *= neutral_curve

        # Reconstruct waveform
        S_flat = mag * np.exp(1j * phase)
        y_out = librosa.istft(S_flat, hop_length=512)
        logging.info("ISTFT and reconstruction complete.")

        # Output .wav to buffer
        out_buffer = io.BytesIO()
        sf.write(out_buffer, y_out, sr, format='WAV')
        out_buffer.seek(0)

        return StreamingResponse(out_buffer, media_type="audio/wav", headers={
            "Content-Disposition": "attachment; filename=flatlined.wav"
        })

    except Exception as e:
        logging.error(f"Error during /neutralize/: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
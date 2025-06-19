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

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize app
app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== /analyze/ ROUTE ==========
class Config(BaseModel):
    k: int
    similarity: Literal["gaussian", "knn", "epsilon"]
    laplacian_type: Literal["unnormalized", "normalized", "random_walk"]

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), config: Config = None):
    df = pd.read_csv(io.StringIO((await file.read()).decode()))
    X = df.values

    # Similarity matrix
    if config.similarity == "gaussian":
        sim = rbf_kernel(X)
    elif config.similarity == "knn":
        sim = kneighbors_graph(X, n_neighbors=10, mode='connectivity').toarray()
    elif config.similarity == "epsilon":
        D = pairwise_distances(X)
        sim = (D < 0.5).astype(float)

    # Laplacian
    norm = config.laplacian_type != "unnormalized"
    lap, _ = csgraph_laplacian(sim, normed=norm, return_diag=True)

    # Eigen decomposition
    eigvals, eigvecs = eigsh(lap, k=config.k + 1, which='SM')
    eigvecs = eigvecs[:, 1:config.k+1]

    # k-means on spectral embeddings
    labels = KMeans(n_clusters=config.k).fit_predict(eigvecs)

    return {
        "clusters": labels.tolist(),
        "eigenvalues": eigvals.tolist()
    }

# ========== /neutralize/ ROUTE ==========
@app.post("/neutralize/")
async def neutralize(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")

    if not file.filename.endswith(".wav"):
        return JSONResponse({"error": "Only .wav files supported"}, status_code=400)

    try:
        # Load and decode WAV
        y, sr = sf.read(io.BytesIO(await file.read()))
        logging.info(f"Sample rate: {sr}, shape: {y.shape}")

        # Convert stereo to mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)
            logging.info("Converted to mono.")

        # Compute STFT
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        mag, phase = np.abs(S), np.angle(S)
        logging.info("Computed STFT.")

        # Average spectrum
        avg_spectrum = mag.mean(axis=1)
        avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum)

        # Invert for flattening
        neutral_curve_db = -avg_spectrum_db + np.max(avg_spectrum_db)
        neutral_curve = librosa.db_to_amplitude(neutral_curve_db)

        # Apply flattening to every frame
        for i in range(S.shape[1]):
            mag[:, i] *= neutral_curve

        S_flat = mag * np.exp(1j * phase)
        y_out = librosa.istft(S_flat, hop_length=512)
        logging.info("Reconstructed time-domain signal.")

        # Output to memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, y_out, sr, format='WAV')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav", headers={
            "Content-Disposition": "attachment; filename=flatlined.wav"
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

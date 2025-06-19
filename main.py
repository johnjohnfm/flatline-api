diff --git a/main.py b/main.py
index ea55f9fe803a5a597bdf11442a2bb4f27a502509..e866709c7cd885ae7325797eb8bd0a929938ebe2 100644
--- a/main.py
+++ b/main.py
@@ -1,124 +1,123 @@
-from fastapi import FastAPI, UploadFile, File
+import os
+import asyncio
+from fastapi import FastAPI, UploadFile, File, Form, HTTPException
 from fastapi.middleware.cors import CORSMiddleware
 from fastapi.responses import JSONResponse, StreamingResponse
-from pydantic import BaseModel
+from pydantic import BaseModel, Field, ValidationError
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
+origins = os.getenv("CORS_ORIGINS", "http://localhost").split(",")
 app.add_middleware(
     CORSMiddleware,
-    allow_origins=["*"],
+    allow_origins=origins,
     allow_methods=["*"],
     allow_headers=["*"],
 )
 
 # -----------------------------
 # /analyze/: Spectral Clustering
 # -----------------------------
 class Config(BaseModel):
-    k: int
+    k: int = Field(..., ge=2, description="Number of clusters")
     similarity: Literal["gaussian", "knn", "epsilon"]
     laplacian_type: Literal["unnormalized", "normalized", "random_walk"]
 
-@app.post("/analyze/")
-async def analyze(file: UploadFile = File(...), config: Config = None):
-    df = pd.read_csv(io.StringIO((await file.read()).decode()))
-    X = df.values
-
-    if config.similarity == "gaussian":
+def _cluster_data(X: np.ndarray, cfg: Config):
+    if cfg.similarity == "gaussian":
         sim = rbf_kernel(X)
-    elif config.similarity == "knn":
-        sim = kneighbors_graph(X, n_neighbors=10, mode='connectivity').toarray()
-    elif config.similarity == "epsilon":
+    elif cfg.similarity == "knn":
+        sim = kneighbors_graph(X, n_neighbors=10, mode="connectivity").toarray()
+    else:
         D = pairwise_distances(X)
         sim = (D < 0.5).astype(float)
 
-    norm = config.laplacian_type != "unnormalized"
+    norm = cfg.laplacian_type != "unnormalized"
     lap, _ = csgraph_laplacian(sim, normed=norm, return_diag=True)
 
-    eigvals, eigvecs = eigsh(lap, k=config.k + 1, which='SM')
-    eigvecs = eigvecs[:, 1:config.k+1]
+    eigvals, eigvecs = eigsh(lap, k=cfg.k + 1, which="SM")
+    eigvecs = eigvecs[:, 1 : cfg.k + 1]
+    labels = KMeans(n_clusters=cfg.k).fit_predict(eigvecs)
+    return labels.tolist(), eigvals.tolist()
+
+
+@app.post("/analyze/")
+async def analyze(file: UploadFile = File(...), config: str = Form(...)):
+    try:
+        cfg = Config.parse_raw(config)
+    except ValidationError as e:
+        raise HTTPException(status_code=400, detail=e.errors())
 
-    labels = KMeans(n_clusters=config.k).fit_predict(eigvecs)
+    df = pd.read_csv(io.StringIO((await file.read()).decode()), header=None)
+    labels, eigvals = await asyncio.to_thread(_cluster_data, df.values, cfg)
 
-    return {
-        "clusters": labels.tolist(),
-        "eigenvalues": eigvals.tolist()
-    }
+    return {"clusters": labels, "eigenvalues": eigvals}
 
 # -----------------------------
 # /neutralize/: Audio Flattening
 # -----------------------------
-@app.post("/neutralize/")
-async def neutralize(file: UploadFile = File(...)):
-    logging.info(f"Received file: {file.filename}")
-
-    if not file.filename.endswith(".wav"):
-        return JSONResponse({"error": "Only .wav files supported"}, status_code=400)
+def _flatten_audio(data: bytes) -> bytes:
+    buffer = io.BytesIO(data)
+    buffer.seek(0)
+    y, sr = sf.read(buffer, dtype="float32")
+    logging.info(f"Sample rate: {sr}, shape: {y.shape}")
 
-    try:
-        # Read and buffer the file
-        data = await file.read()
-        buffer = io.BytesIO(data)
-        buffer.seek(0)
+    if len(y.shape) > 1:
+        y = y.mean(axis=1)
+        logging.info("Converted to mono.")
 
-        # Load with soundfile (preserves 32-bit float and stereo)
-        y, sr = sf.read(buffer, dtype='float32')
-        logging.info(f"Sample rate: {sr}, shape: {y.shape}")
+    S = librosa.stft(y, n_fft=2048, hop_length=512)
+    mag, phase = np.abs(S), np.angle(S)
 
-        # Optional: convert to mono
-        if len(y.shape) > 1:
-            y = y.mean(axis=1)
-            logging.info("Converted to mono.")
+    avg_spectrum = mag.mean(axis=1)
+    avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum)
+    neutral_curve_db = -avg_spectrum_db + np.max(avg_spectrum_db)
+    neutral_curve = librosa.db_to_amplitude(neutral_curve_db)
 
-        # Compute STFT
-        S = librosa.stft(y, n_fft=2048, hop_length=512)
-        mag, phase = np.abs(S), np.angle(S)
-        logging.info("STFT complete.")
+    for i in range(mag.shape[1]):
+        mag[:, i] *= neutral_curve
 
-        # Compute average spectrum
-        avg_spectrum = mag.mean(axis=1)
-        avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum)
+    S_flat = mag * np.exp(1j * phase)
+    y_out = librosa.istft(S_flat, hop_length=512)
 
-        # Create neutralizing curve
-        neutral_curve_db = -avg_spectrum_db + np.max(avg_spectrum_db)
-        neutral_curve = librosa.db_to_amplitude(neutral_curve_db)
+    out_buffer = io.BytesIO()
+    sf.write(out_buffer, y_out, sr, format="WAV")
+    out_buffer.seek(0)
+    return out_buffer.getvalue()
 
-        # Apply neutralizing curve
-        for i in range(mag.shape[1]):
-            mag[:, i] *= neutral_curve
 
-        # Reconstruct waveform
-        S_flat = mag * np.exp(1j * phase)
-        y_out = librosa.istft(S_flat, hop_length=512)
-        logging.info("ISTFT and reconstruction complete.")
-
-        # Output .wav to buffer
-        out_buffer = io.BytesIO()
-        sf.write(out_buffer, y_out, sr, format='WAV')
-        out_buffer.seek(0)
+@app.post("/neutralize/")
+async def neutralize(file: UploadFile = File(...)):
+    logging.info(f"Received file: {file.filename}")
 
-        return StreamingResponse(out_buffer, media_type="audio/wav", headers={
-            "Content-Disposition": "attachment; filename=flatlined.wav"
-        })
+    if not file.filename.endswith(".wav"):
+        raise HTTPException(status_code=400, detail="Only .wav files supported")
 
+    try:
+        data = await file.read()
+        out_bytes = await asyncio.to_thread(_flatten_audio, data)
+        return StreamingResponse(
+            io.BytesIO(out_bytes),
+            media_type="audio/wav",
+            headers={"Content-Disposition": "attachment; filename=flatlined.wav"},
+        )
     except Exception as e:
         logging.error(f"Error during /neutralize/: {e}")
-        return JSONResponse({"error": str(e)}, status_code=500)
+        raise HTTPException(status_code=500, detail=str(e))

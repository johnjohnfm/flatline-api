from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    eigvecs = eigvecs[:, 1:config.k+1]  # Skip first

    # k-means on spectral embedding
    labels = KMeans(n_clusters=config.k).fit_predict(eigvecs)

    return {
        "clusters": labels.tolist(),
        "eigenvalues": eigvals.tolist()
    }

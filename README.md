diff --git a//dev/null b/README.md
index 0000000000000000000000000000000000000000..bf0c608e2b70f7b0072df96ba9383cb00889739c 100644
--- a//dev/null
+++ b/README.md
@@ -0,0 +1,48 @@
+# Flatline API
+
+A small FastAPI service providing two endpoints:
+
+- `/analyze/` - perform spectral clustering on an uploaded CSV file using customizable parameters.
+- `/neutralize/` - flatten the spectrum of an uploaded WAV file.
+
+## Setup
+
+1. Create and activate a virtual environment.
+2. Install dependencies:
+   ```bash
+   pip install -r requirements.txt
+   ```
+3. Run the development server:
+   ```bash
+   uvicorn main:app --reload
+   ```
+
+The service uses CORS origins defined via the `CORS_ORIGINS` environment variable (comma separated). In development it defaults to `http://localhost`.
+
+## Usage
+
+### Analyze
+Send a multipart request with a CSV file and a JSON configuration. Example using `curl`:
+
+```bash
+curl -F "file=@data.csv" \
+     -F "config={\"k\":2,\"similarity\":\"gaussian\",\"laplacian_type\":\"normalized\"}" \
+     http://localhost:8000/analyze/
+```
+
+### Neutralize
+Upload a WAV file and receive a flattened WAV in response:
+
+```bash
+curl -F "file=@audio.wav" http://localhost:8000/neutralize/ -o flatlined.wav
+```
+
+## Testing
+
+Run the test suite with:
+
+```bash
+pytest
+```
+
+

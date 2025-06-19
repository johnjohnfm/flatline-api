diff --git a//dev/null b/tests/test_main.py
index 0000000000000000000000000000000000000000..f3a45cd4d18927d64e60d88d7b2082e569b364a2 100644
--- a//dev/null
+++ b/tests/test_main.py
@@ -0,0 +1,39 @@
+import json
+import os
+import sys
+import numpy as np
+import soundfile as sf
+sys.path.append(os.path.dirname(os.path.dirname(__file__)))
+from fastapi.testclient import TestClient
+from main import app
+
+client = TestClient(app)
+
+def test_analyze_gaussian(tmp_path):
+    data = np.array([[0,0],[1,1],[2,2]], dtype=float)
+    csv_path = tmp_path / "data.csv"
+    np.savetxt(csv_path, data, delimiter=",")
+
+    config = {"k":2,"similarity":"gaussian","laplacian_type":"normalized"}
+    with open(csv_path, "rb") as f:
+        response = client.post(
+            "/analyze/",
+            files={"file": ("data.csv", f, "text/csv")},
+            data={"config": json.dumps(config)},
+        )
+    assert response.status_code == 200
+    payload = response.json()
+    assert "clusters" in payload
+    assert len(payload["clusters"]) == data.shape[0]
+
+def test_neutralize(tmp_path):
+    sr = 22050
+    y = np.zeros(sr, dtype=np.float32)
+    wav_path = tmp_path / "tone.wav"
+    sf.write(wav_path, y, sr)
+
+    with open(wav_path, "rb") as f:
+        response = client.post("/neutralize/", files={"file": ("tone.wav", f, "audio/wav")})
+    assert response.status_code == 200
+    assert response.headers["content-type"] == "audio/wav"
+

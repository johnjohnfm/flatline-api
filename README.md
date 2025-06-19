# FLATLINE WEB API

**FLATLINE WEB API** is a high-performance web-based audio analysis tool designed for real-time spectral diagnostics, flattening, and normalization. Ideal for engineers, developers, and audio brands, it enables web-embeddable access to frequency response data, loudness profiles, and spectral clustering—all without leaving the browser.

---

## 🔧 Features

- 🎚️ **Spectral Flattening** — Neutralize tonal coloration by matching to a flat frequency curve
- 📈 **Frequency Response Analysis** — Visualize and extract the spectral fingerprint of any audio file
- 🔊 **Loudness Normalization** — Match track levels to LUFS standards for consistent output
- 🧠 **Spectral Clustering** — Group similar sonic profiles using ML-driven clustering
- 💻 **Embeddable Widget** — Add powerful audio analytics directly to any site or product
- ⚡ **Fast + Modular** — Optimized for speed, designed for extensibility

---

## 🚀 Usage

1. **Upload Audio** (WAV or MP3)
2. **Receive Analysis** — JSON output includes:
   - Full-band spectral data
   - Target deviation curve
   - LUFS value and normalization suggestion
   - Optional spectral cluster ID

3. **Optional**: Apply spectral flattening via API call to generate a flat-reference version

---

## 📦 API Endpoints (Beta)

```http
POST /analyze

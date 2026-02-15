# South Sudan Cattle Movement Prediction System

A web app that predicts where cattle are likely to be in South Sudan, updated hourly, accessible via a simple shareable link. Built for humanitarian and planning use (e.g. UN/NGOs).

**For developers:** See [API Reference](#api-reference-for-developers) for endpoints and response formats, [Project Structure](#project-structure) for the codebase layout, and [Deployment (Railway)](#chunk-7-deployment-railway) for production setup.

## Quick Start

1. **Clone and install dependencies:**
   ```bash
   cd southsudan-cattle
   pip install -r requirements.txt
   ```

2. **Set up Google Earth Engine:**
   - Sign up at https://earthengine.google.com/signup/
   - Run `earthengine authenticate` for interactive auth
   - Or set `GOOGLE_EARTH_ENGINE_KEY` for service account

3. **Run the data pipeline (Chunk 1):**
   ```bash
   python pipeline/data_pipeline.py
   ```
   The pipeline processes **one week (one chunk) at a time** and saves to disk after each chunk to avoid high memory use and crashes on 16GB RAM systems. Do not load all 26 weeks into RAM at once.

4. **Start the Flask app (Chunk 5):**
   ```bash
   python app/app.py
   ```
   Then open http://localhost:5000

   Endpoints: `GET /` (map), `GET /api/predictions` (GeoJSON), `GET /api/status`, `GET /about`.

## Chunk 6: Frontend map

- **Map:** Leaflet, centered on South Sudan, OpenStreetMap tiles, GeoJSON layer with probability colors (green → yellow → red).
- **Clickable cells:** Popup shows cell ID, probability %, last satellite update.
- **Info panel (top-right):** Last/next update, trend, confidence.
- **Legend (bottom-left):** Low–High color scale.
- **Auto-refresh:** Predictions reload every hour.
- **Mobile:** Map fills viewport; info panel collapses to an accordion (toggle) on small screens.

## Project Structure

```
southsudan-cattle/
├── data/
│   ├── raw/           # Downloaded satellite data
│   ├── processed/     # Processed grid features
│   └── baseline/      # FAO cattle density, FEWS NET shapefiles
├── models/
│   ├── cattle_model.pth
│   └── architecture.py
├── pipeline/
│   ├── data_pipeline.py
│   ├── train.py
│   └── predict.py
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── outputs/
├── requirements.txt
└── README.md
```

## Chunk 4: Hourly prediction

- **Run once:** `python pipeline/predict.py`  
  Writes `outputs/latest_prediction.geojson` (FeatureCollection with probability per cell, confidence, last_satellite_update).
- **Run every hour:** `python pipeline/predict.py --schedule` (uses APScheduler).
- **Caching:** `outputs/last_prediction_state.npz` stores centroid and timestamp for movement direction and decay.
- **Weather:** Set `OPENWEATHER_API_KEY` for rainfall/temperature adjustments; optional.

## Chunk 3: Model training

- **Train:** `python pipeline/train.py [--epochs 10] [--batch-size 4]`
- Uses mixed precision (AMP) by default; `--no-amp` to disable. Saves `checkpoint_epoch_N.pth` every epoch and `cattle_model.pth` / `cattle_model_best.pth` at the end.
- Loss: BCEWithLogits + NDVI penalty (dry cells) + recency penalty (recently grazed). Movement constraint is applied at inference (Chunk 4).

## Chunk 2: Model architecture (U-Net + LSTM)

- **GPU:** PyTorch with CUDA is recommended. Install with CUDA 11.8:  
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- **RTX 3060:** Batch size is fixed at 4; data loaders use `num_workers=2` and `pin_memory=True` when CUDA is available.
- **Test forward pass and GPU memory:**  
  `python models/test_forward.py`  
  Uses dummy shape `(4, 4, 7, 50, 50)` and checks that GPU memory stays under 6 GB. Also runs a mixed-precision forward pass.

## Memory safety (16GB RAM / RTX 3060)

- The data pipeline saves **one chunk at a time**: it processes one week of satellite data, writes that week’s features and labels to disk, then frees memory before the next week. This avoids loading all 26 weeks into RAM and prevents crashes.
- To verify outputs without loading full arrays into RAM: `python pipeline/data_pipeline.py --verify --mmap`

## Chunk 7: Deployment (Railway)

1. **Create a Railway project** and connect your repo (or deploy from CLI).

2. **Set environment variables** in the Railway dashboard:
   - `PORT` – Set automatically by Railway; the app reads it.
   - `OPENWEATHER_API_KEY` – Optional; for hourly weather adjustments.
   - `GOOGLE_EARTH_ENGINE_KEY` – Optional for pipeline; only needed if you run data pipeline on Railway.
   - `MODEL_VERSION` – Optional; e.g. `v1.0` (default).

3. **Start command:** Railway will use the **Procfile**: `web: gunicorn --bind 0.0.0.0:$PORT app.app:app`. The app listens on `0.0.0.0` and the port Railway provides.

4. **Persistent storage:** For production, attach a **Railway volume** and mount it so that `outputs/` (and optionally `data/processed/`, `models/`) are on the volume. Otherwise predictions and model are ephemeral. You can also commit a pre-built `cattle_model.pth` and `outputs/latest_prediction.geojson` to the repo so the first deploy serves a map without running the pipeline.

5. **Automated tasks:**
   - **Hourly (predictions):** Use Railway **Cron** or an external cron to run every hour:
     ```bash
     python pipeline/predict.py
     ```
     (Run from project root; ensure `outputs/` is writable and on a volume if you want persistence.)
   - **Weekly (satellite data):** Optionally run weekly:
     ```bash
     python pipeline/data_pipeline.py
     ```
     Requires Google Earth Engine credentials and sufficient memory.

6. **Monitoring:** Call `GET /api/health` for a simple health check (`{"status": "ok", "model_version": "..."}`). Use it for uptime checks or load balancers.

7. **API endpoints (for reference):**
   - `GET /` – Map page.
   - `GET /api/predictions` – GeoJSON of current predictions.
   - `GET /api/status` – Last update, next update, model version, satellite freshness.
   - `GET /api/health` – Health check for monitoring.
   - `GET /about` – Methodology page.

8. **Test:** Open the generated Railway URL; the map should load. Share the link to confirm others can open it.

## Environment Variables

- `PORT` – Set by Railway (or use 5000 locally).
- `GOOGLE_EARTH_ENGINE_KEY` – Path to GEE service account JSON (or use `earthengine authenticate`).
- `OPENWEATHER_API_KEY` – For hourly weather-based adjustments.
- `MODEL_VERSION` – Optional; reported in `/api/status` and `/api/health`.

## API Reference (for developers)

All responses are JSON unless noted. The web app serves HTML at `GET /` and `GET /about`.

| Endpoint | Description | Response |
|----------|-------------|----------|
| `GET /` | Map page (HTML) | Renders `index.html` |
| `GET /about` | Methodology page (HTML) | Renders `about.html` |
| `GET /api/predictions` | Latest prediction GeoJSON | `{ "type": "FeatureCollection", "timestamp": "...", "features": [ { "type": "Feature", "geometry": {...}, "properties": { "cell_id": "cell_0", "probability": 0.73, "confidence": "medium", "last_satellite_update": "2024-02-06" } }, ... ] }` |
| `GET /api/status` | System status | `{ "last_update": "2024-02-09T14:00:00Z", "next_update": "2024-02-09T15:00:00Z", "model_version": "v1.0", "satellite_freshness": "3 days" }` |
| `GET /api/health` | Health check | `{ "status": "ok", "model_version": "v1.0" }` |

- **404** on `/api/predictions` when no prediction file exists (e.g. before first run of `pipeline/predict.py`).
- **500** on read/parse errors; see server logs.

## Data Sources

- **Sentinel-2** – NDVI (vegetation): [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- **CHIRPS** – Rainfall: [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
- **Sentinel-1** – Soil moisture (radar): [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- **ERA5** – Temperature: [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY)
- **FAO Gridded Livestock of the World** – Cattle density (training): [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GIVQ75)
- **OpenWeather** – Current weather (hourly adjustments): [openweathermap.org](https://openweathermap.org/api)

## Chunk 8: About page and documentation

- **About page** (`GET /about`): Explains how the system works, what the probabilities mean, update frequency, data sources (with links), limitations, disclaimer, and contact/feedback. A non-technical reader can understand the methodology from this page.
- **API reference:** Documented above with endpoints and response formats.
- **Data sources:** Listed with links in this README and on the about page.
- **Code:** Pipeline and app modules include docstrings and comments for key logic.

## License

Open source. Suitable for UN/NGO humanitarian use.

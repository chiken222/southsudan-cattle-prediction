"""
Chunk 4: Hourly prediction system.
- Loads trained model, runs inference with torch.no_grad()
- Applies NDVI and movement constraints
- Fetches OpenWeather for South Sudan, applies hourly adjustments
- Computes movement direction (centroid shift), confidence level
- Writes GeoJSON to outputs/latest_prediction.geojson
- Supports one-off run or APScheduler hourly job with caching
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import shape, Point
# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    SOUTH_SUDAN_BBOX,
    DEG_PER_10KM_LAT,
    DEG_PER_10KM_LON,
)
from models.architecture import create_model
from models.dataloader import load_processed_data, reshape_for_model
from models.gpu_utils import get_device
from models.losses import apply_ndvi_constraint, apply_movement_constraint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Hourly adjustment constants (from spec)
RAIN_BOOST_NEAR_WATER = 0.1       # if rainfall_last_hour > 5mm
RAIN_THRESHOLD_MM = 5.0
HOT_TEMP_C = 35.0
HOT_HOURS_START, HOT_HOURS_END = 12, 15  # 12:00-15:00 local
DECAY_PER_24H = 0.9              # probability_previous *= 0.9 if > 24h since pred
MAX_MOVEMENT_CELLS = 5
PROB_CLIP_MIN, PROB_CLIP_MAX = 0.1, 0.9  # avoid 0% or 100% claims

# South Sudan center for weather API
SS_CENTER_LAT = 8.0
SS_CENTER_LON = 30.0


def get_cell_geometry(row: int, col: int, n_rows: int, n_cols: int) -> list:
    """Return GeoJSON polygon coordinates for cell (row, col). Row 0 = south."""
    south = SOUTH_SUDAN_BBOX["south"]
    west = SOUTH_SUDAN_BBOX["west"]
    lat_s = south + row * DEG_PER_10KM_LAT
    lat_n = south + (row + 1) * DEG_PER_10KM_LAT
    lon_w = west + col * DEG_PER_10KM_LON
    lon_e = west + (col + 1) * DEG_PER_10KM_LON
    # GeoJSON: [lon, lat], closed ring
    coords = [
        [lon_w, lat_s], [lon_e, lat_s], [lon_e, lat_n], [lon_w, lat_n], [lon_w, lat_s],
    ]
    return [coords]


def fetch_weather(api_key: str) -> dict:
    """Fetch current weather for South Sudan from OpenWeather. Returns dict or empty on failure."""
    if not api_key:
        return {}
    try:
        import requests
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={SS_CENTER_LAT}&lon={SS_CENTER_LON}&appid={api_key}&units=metric"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("OpenWeather fetch failed: %s", e)
        return {}


def hourly_adjustments(
    prob_grid: np.ndarray,
    weather: dict,
    hours_since_last_pred: float,
    water_dist_grid: np.ndarray,
    utc_hour: int,
) -> np.ndarray:
    """
    Apply hourly adjustment rules to probability grid (n_rows, n_cols).
    - Rainfall > 5mm: boost near water
    - Temp > 35°C and 12-15h local: scale down (cattle seek shade)
    - Decay if > 24h since last prediction
    """
    out = np.clip(prob_grid.astype(np.float64), PROB_CLIP_MIN, PROB_CLIP_MAX).copy()

    # Rainfall: boost cells near water (low water_dist) if recent rain
    rain_1h = weather.get("rain", {}).get("1h") or 0
    if rain_1h > RAIN_THRESHOLD_MM and water_dist_grid.size > 0:
        near_water = water_dist_grid < 20  # km
        out[near_water] = np.minimum(1.0, out[near_water] + RAIN_BOOST_NEAR_WATER)

    # Hot midday: scale down (approximate local = UTC+2 for South Sudan)
    local_hour = (utc_hour + 2) % 24
    temp = weather.get("main", {}).get("temp")
    if temp is not None and temp > HOT_TEMP_C and HOT_HOURS_START <= local_hour <= HOT_HOURS_END:
        out *= 0.7

    # Decay if stale
    if hours_since_last_pred > 24:
        out *= DECAY_PER_24H

    return np.clip(out, PROB_CLIP_MIN, PROB_CLIP_MAX).astype(np.float32)


def weighted_centroid(prob_grid: np.ndarray, n_rows: int, n_cols: int) -> tuple:
    """Return (lat, lon) of probability-weighted centroid. Row 0 = south."""
    south, west = SOUTH_SUDAN_BBOX["south"], SOUTH_SUDAN_BBOX["west"]
    total = float(np.sum(prob_grid)) + 1e-9
    lat_sum = lon_sum = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            w = prob_grid[r, c]
            lat_sum += w * (south + (r + 0.5) * DEG_PER_10KM_LAT)
            lon_sum += w * (west + (c + 0.5) * DEG_PER_10KM_LON)
    return (lat_sum / total, lon_sum / total)


def compute_confidence(last_satellite_days_ago: float) -> str:
    """Return 'low' | 'medium' | 'high' based on satellite data freshness."""
    if last_satellite_days_ago <= 3:
        return "high"
    if last_satellite_days_ago <= 7:
        return "medium"
    return "low"


def run_inference(
    model_path: Path = None,
    data_dir: Path = None,
    out_dir: Path = None,
    openweather_api_key: str = None,
    last_state_path: Path = None,
) -> dict:
    """
    Run one prediction: load model and latest features, infer, apply weather, write GeoJSON.
    Returns summary dict with timestamp, movement_direction, confidence, etc.
    """
    if model_path is None:
        model_path = MODELS_DIR / "cattle_model.pth"
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR
    if out_dir is None:
        out_dir = OUTPUTS_DIR
    if last_state_path is None:
        last_state_path = out_dir / "last_prediction_state.npz"

    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    now = datetime.now(timezone.utc)

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        n_rows = ckpt.get("n_rows")
        n_cols = ckpt.get("n_cols")
        seq_len = ckpt.get("seq_len", 4)
    else:
        state = ckpt
        n_rows = n_cols = seq_len = None

    if n_rows is None or n_cols is None:
        # Fallback from processed data
        with open(data_dir / "cell_metadata.json") as f:
            meta = json.load(f)
        n_rows, n_cols = meta["n_rows"], meta["n_cols"]
        seq_len = seq_len or 4

    model = create_model(
        num_features=7,
        num_weeks=seq_len,
        grid_height=n_rows,
        grid_width=n_cols,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    # Load latest features (last seq_len weeks)
    features, _, meta = load_processed_data(data_dir)
    feat_grid, _ = reshape_for_model(features, np.zeros_like(features[:, :, 0]), meta)
    num_weeks = feat_grid.shape[0]
    if num_weeks < seq_len:
        # Pad with last week repeated
        pad = np.repeat(feat_grid[-1:], seq_len - num_weeks, axis=0)
        feat_grid = np.concatenate([pad, feat_grid], axis=0)
    seq = feat_grid[-seq_len:]  # (T, C, H, W)
    seq_t = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)

    # Inference with no_grad
    with torch.no_grad():
        logits = model(seq_t)
        pred = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

    ndvi_last = torch.from_numpy(seq[-1:].astype(np.float32)[:, 0:1].copy()).to(device)  # (1, 1, H, W)
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
    pred_t = apply_ndvi_constraint(pred_t, ndvi_last)
    prev_high = (seq[-1, 4] > 0.5)
    prev_high_t = torch.from_numpy(prev_high.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    pred_t = apply_movement_constraint(pred_t, prev_high_t, MAX_MOVEMENT_CELLS)
    prob_grid = pred_t.squeeze().cpu().numpy()

    # Hourly adjustments
    weather = fetch_weather(openweather_api_key or os.environ.get("OPENWEATHER_API_KEY", ""))
    water_dist = seq[-1, 5]  # (H, W)
    hours_since = 1.0  # assume hourly run
    if last_state_path.exists():
        try:
            prev = np.load(last_state_path)
            prev_ts = float(prev.get("timestamp", 0))
            hours_since = max(0.1, (now.timestamp() - prev_ts) / 3600)
        except Exception:
            pass
    prob_grid = hourly_adjustments(
        prob_grid, weather,
        hours_since_last_pred=hours_since,
        water_dist_grid=water_dist,
        utc_hour=now.hour,
    )

    # Movement direction from previous centroid
    movement_direction = None
    if last_state_path.exists():
        try:
            prev = np.load(last_state_path)
            prev_lat = float(prev["centroid_lat"])
            prev_lon = float(prev["centroid_lon"])
            cur_lat, cur_lon = weighted_centroid(prob_grid, n_rows, n_cols)
            dlat = cur_lat - prev_lat
            dlon = cur_lon - prev_lon
            if abs(dlon) > 1e-5 or abs(dlat) > 1e-5:
                direction = "NE" if dlon >= 0 and dlat >= 0 else "NW" if dlon < 0 and dlat >= 0 else "SE" if dlon >= 0 else "SW"
                movement_direction = f"Moving {direction}"
        except Exception:
            movement_direction = "Unknown"

    cur_lat, cur_lon = weighted_centroid(prob_grid, n_rows, n_cols)
    created_at = meta.get("created_at", "")
    try:
        # Parse ISO timestamp (e.g. 2026-02-10T02:22:53.715815Z)
        ts = created_at.replace("Z", "+00:00")
        sat_dt = datetime.fromisoformat(ts)
        if sat_dt.tzinfo is None:
            sat_dt = sat_dt.replace(tzinfo=timezone.utc)
        last_satellite_days = (now - sat_dt).total_seconds() / 86400
    except Exception:
        last_satellite_days = 7.0
    confidence = compute_confidence(last_satellite_days)

    # Build GeoJSON
    features_geojson = []
    for r in range(n_rows):
        for c in range(n_cols):
            cell_idx = r * n_cols + c
            prob = float(prob_grid[r, c])
            features_geojson.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": get_cell_geometry(r, c, n_rows, n_cols)},
                "properties": {
                    "cell_id": f"cell_{cell_idx}",
                    "probability": round(prob, 4),
                    "confidence": confidence,
                    "last_satellite_update": created_at[:10] if len(created_at) >= 10 else created_at,
                },
            })
# Filter features to only show cells inside South Sudan
    try:
        border_path = Path(__file__).parent.parent / 'data' / 'south_sudan_only.geojson'
        with open(border_path) as f:
            ss_data = json.load(f)
            ss_polygon = shape(ss_data['features'][0]['geometry'])
        
        # Filter features
        filtered_features = []
        for feat in features_geojson:
            # Get center point of cell
            coords = feat['geometry']['coordinates'][0]
            center_lon = sum(c[0] for c in coords) / len(coords)
            center_lat = sum(c[1] for c in coords) / len(coords)
            cell_center = Point(center_lon, center_lat)
            
            if ss_polygon.contains(cell_center):
                filtered_features.append(feat)
        
        features_geojson = filtered_features
        logger.info("Filtered to %d cells inside South Sudan", len(filtered_features))
    except Exception as e:
        logger.warning("Could not filter by border: %s", e)
    geojson = {
        "type": "FeatureCollection",
        "timestamp": now.isoformat(),
        "next_update": now.replace(hour=now.hour + 1).isoformat() if now.hour < 23 else now.replace(hour=0).isoformat(),
        "features": features_geojson,
        "meta": {
            "movement_direction": movement_direction,
            "confidence": confidence,
            "last_satellite_days": round(last_satellite_days, 1),
        },
    }

    out_path = out_dir / "latest_prediction.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Cache state for next run (centroid + prob for decay)
    np.savez(
        last_state_path,
        prob_grid=prob_grid,
        centroid_lat=cur_lat,
        centroid_lon=cur_lon,
        timestamp=now.timestamp(),
    )

    return {
        "timestamp": now.isoformat(),
        "path": str(out_path),
        "confidence": confidence,
        "movement_direction": movement_direction,
        "n_features": len(features_geojson),
    }


def schedule_hourly(
    model_path: Path = None,
    data_dir: Path = None,
    out_dir: Path = None,
    openweather_api_key: str = None,
):
    """Run prediction every hour via APScheduler. Caching is inside run_inference."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logger.error("Install apscheduler: pip install apscheduler")
        return
    scheduler = BlockingScheduler()

    def job():
        run_inference(
            model_path=model_path or MODELS_DIR / "cattle_model.pth",
            data_dir=data_dir or PROCESSED_DATA_DIR,
            out_dir=out_dir or OUTPUTS_DIR,
            openweather_api_key=openweather_api_key or os.environ.get("OPENWEATHER_API_KEY"),
        )

    scheduler.add_job(job, "interval", hours=1, id="cattle_predict")
    job()  # run once immediately
    logger.info("Scheduler started: running every hour")
    scheduler.start()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run hourly cattle prediction (Chunk 4)")
    p.add_argument("--model", type=Path, default=MODELS_DIR / "cattle_model.pth")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    p.add_argument("--openweather-key", default=os.environ.get("OPENWEATHER_API_KEY"))
    p.add_argument("--schedule", action="store_true", help="Run scheduler (hourly updates)")
    args = p.parse_args()
    if args.schedule:
        schedule_hourly(
            model_path=args.model,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            openweather_api_key=args.openweather_key,
        )
        return 0
    result = run_inference(
        model_path=args.model,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        openweather_api_key=args.openweather_key,
    )
    logger.info("Result: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

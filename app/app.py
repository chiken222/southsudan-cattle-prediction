"""
Chunk 5: Flask backend for South Sudan cattle movement prediction.
Serves main page, API predictions, status, and about page.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Project root so config can be imported when running from app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template

from config import OUTPUTS_DIR

app = Flask(
    __name__,
    template_folder=Path(__file__).parent / "templates",
    static_folder=Path(__file__).parent / "static",
)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_FILE = OUTPUTS_DIR / "latest_prediction.geojson"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")


def _after_request(response):
    """Add CORS headers for API access from other origins."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


app.after_request(_after_request)


@app.route("/")
def index():
    """Serve main map page."""
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    """Return latest GeoJSON with probabilities per grid cell."""
    if not PREDICTION_FILE.exists():
        logger.warning("Prediction file not found: %s", PREDICTION_FILE)
        return jsonify({"error": "No predictions available yet. Run pipeline/predict.py first."}), 404
    try:
        with open(PREDICTION_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, OSError) as e:
        logger.exception("Failed to load predictions: %s", e)
        return jsonify({"error": "Failed to load prediction data."}), 500


@app.route("/api/status")
def api_status():
    """Return system status: last_update, next_update, model_version, satellite_freshness."""
    if not PREDICTION_FILE.exists():
        return jsonify({
            "last_update": None,
            "next_update": None,
            "model_version": MODEL_VERSION,
            "satellite_freshness": None,
            "message": "No prediction file yet. Run pipeline/predict.py.",
        })
    try:
        with open(PREDICTION_FILE, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta", {})
        return jsonify({
            "last_update": data.get("timestamp"),
            "next_update": data.get("next_update"),
            "model_version": MODEL_VERSION,
            "satellite_freshness": f"{meta.get('last_satellite_days', '?')} days",
        })
    except (json.JSONDecodeError, OSError) as e:
        logger.exception("Failed to load status: %s", e)
        return jsonify({"error": "Failed to load status."}), 500


@app.route("/about")
def about():
    """Serve methodology / about page."""
    return render_template("about.html")


@app.route("/api/health")
def api_health():
    """Health check for monitoring and load balancers."""
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting server on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")

"""
Satellite data fetching via Google Earth Engine.
Pulls Sentinel-2 (NDVI), CHIRPS (rainfall), Sentinel-1 (moisture), ERA5 (temp).
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SOUTH_SUDAN_BBOX,
    GEE_DATASETS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def _get_gee_region():
    """Build GEE geometry for South Sudan."""
    try:
        import ee
    except ImportError:
        raise ImportError("Install earthengine-api: pip install earthengine-api")

    # GEE uses [lon, lat] for coordinates
    coords = [
        [SOUTH_SUDAN_BBOX["west"], SOUTH_SUDAN_BBOX["south"]],
        [SOUTH_SUDAN_BBOX["east"], SOUTH_SUDAN_BBOX["south"]],
        [SOUTH_SUDAN_BBOX["east"], SOUTH_SUDAN_BBOX["north"]],
        [SOUTH_SUDAN_BBOX["west"], SOUTH_SUDAN_BBOX["north"]],
    ]
    return ee.Geometry.Polygon(coords)


def _init_gee():
    """Initialize Google Earth Engine."""
    try:
        import ee
    except ImportError:
        raise ImportError("Install earthengine-api: pip install earthengine-api")

    try:
        ee.Initialize(project='astral-reef-472821-f9')
    except Exception as e:
        print("Google Earth Engine not authenticated.")
        print("Run: earthengine authenticate")
        raise e

    return ee


def pull_sentinel2_ndvi(
    start_date: str,
    end_date: str,
    region=None,
) -> Optional[Dict[str, Any]]:
    """
    Pull Sentinel-2 NDVI for South Sudan.
    Returns mean NDVI as reduced image, or None if GEE unavailable.
    """
    try:
        ee = _init_gee()
    except Exception as e:
        print(f"GEE init failed: {e}")
        return None

    if region is None:
        region = _get_gee_region()

    collection = (
        ee.ImageCollection(GEE_DATASETS["sentinel2"])
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def add_ndvi(image):
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return image.addBands(ndvi)

    ndvi_collection = collection.map(add_ndvi)
    ndvi_mean = ndvi_collection.select("NDVI").mean().clip(region)

    # Get reduced region mean (for quick export/use)
    # Scale in meters - 10km = 10000m for our grid
    scale = 10000
    try:
        reduced = ndvi_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        )
        result = reduced.getInfo()
        if result:
            return {"ndvi_mean": result.get("NDVI", 0.3), "source": "sentinel2"}
    except Exception as e:
        print(f"Sentinel-2 reduction failed: {e}")

    return {"ndvi_mean": 0.3, "source": "sentinel2_fallback"}


def pull_chirps_rainfall(
    start_date: str,
    end_date: str,
    region=None,
) -> Optional[Dict[str, Any]]:
    """Pull CHIRPS rainfall for South Sudan. Returns weekly sum in mm."""
    try:
        ee = _init_gee()
    except Exception as e:
        print(f"GEE init failed: {e}")
        return None

    if region is None:
        region = _get_gee_region()

    collection = (
        ee.ImageCollection(GEE_DATASETS["chirps"])
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    rainfall_sum = collection.sum().clip(region)
    scale = 10000
    try:
        reduced = rainfall_sum.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        )
        result = reduced.getInfo()
        if result:
            return {"rainfall_mm": result.get("precipitation", 0), "source": "chirps"}
    except Exception as e:
        print(f"CHIRPS reduction failed: {e}")

    return {"rainfall_mm": 0, "source": "chirps_fallback"}


def pull_sentinel1_moisture(
    start_date: str,
    end_date: str,
    region=None,
) -> Optional[Dict[str, Any]]:
    """Pull Sentinel-1 VV backscatter (proxy for moisture)."""
    try:
        ee = _init_gee()
    except Exception as e:
        print(f"GEE init failed: {e}")
        return None

    if region is None:
        region = _get_gee_region()

    collection = (
        ee.ImageCollection(GEE_DATASETS["sentinel1"])
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .select("VV")
    )

    moisture_mean = collection.mean().clip(region)
    scale = 10000
    try:
        reduced = moisture_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        )
        result = reduced.getInfo()
        if result:
            return {"moisture_vv": result.get("VV", -12), "source": "sentinel1"}
    except Exception as e:
        print(f"Sentinel-1 reduction failed: {e}")

    return {"moisture_vv": -12, "source": "sentinel1_fallback"}


def pull_era5_temperature(
    start_date: str,
    end_date: str,
    region=None,
) -> Optional[Dict[str, Any]]:
    """Pull ERA5 2m temperature for South Sudan."""
    try:
        ee = _init_gee()
    except Exception as e:
        print(f"GEE init failed: {e}")
        return None

    if region is None:
        region = _get_gee_region()

    collection = (
        ee.ImageCollection(GEE_DATASETS["era5"])
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select("mean_2m_air_temperature")
    )

    temp_mean = collection.mean().clip(region)
    scale = 30000  # ERA5 is 30km
    try:
        reduced = temp_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        )
        result = reduced.getInfo()
        if result:
            kelvin = result.get("mean_2m_air_temperature", 300)
            celsius = kelvin - 273.15 if kelvin else 28
            return {"temperature_c": celsius, "source": "era5"}
    except Exception as e:
        print(f"ERA5 reduction failed: {e}")

    return {"temperature_c": 28, "source": "era5_fallback"}


def pull_weekly_satellite_data(weeks_ago: int = 0) -> Dict[str, Any]:
    """
    Pull all satellite data for a given week.
    weeks_ago=0 means last 7 days.
    weeks_ago=1 means 7-14 days ago, etc.
    """
    end = datetime.utcnow() - timedelta(days=weeks_ago * 7)
    start = end - timedelta(days=7)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    ndvi = pull_sentinel2_ndvi(start_str, end_str)
    rainfall = pull_chirps_rainfall(start_str, end_str)
    moisture = pull_sentinel1_moisture(start_str, end_str)
    temp = pull_era5_temperature(start_str, end_str)

    return {
        "start_date": start_str,
        "end_date": end_str,
        "ndvi": ndvi or {"ndvi_mean": 0.3, "source": "fallback"},
        "rainfall": rainfall or {"rainfall_mm": 0, "source": "fallback"},
        "moisture": moisture or {"moisture_vv": -12, "source": "fallback"},
        "temperature": temp or {"temperature_c": 28, "source": "fallback"},
    }


def get_synthetic_weekly_data(week_idx: int) -> Dict[str, Any]:
    """
    Generate synthetic data when GEE is unavailable (for testing).
    Uses reasonable ranges for South Sudan.
    """
    import random
    random.seed(week_idx)
    return {
        "start_date": f"2024-01-{(week_idx * 7) % 28 + 1:02d}",
        "end_date": f"2024-01-{(week_idx * 7 + 7) % 28 + 1:02d}",
        "ndvi": {"ndvi_mean": 0.2 + random.random() * 0.5, "source": "synthetic"},
        "rainfall": {"rainfall_mm": random.random() * 50, "source": "synthetic"},
        "moisture": {"moisture_vv": -15 + random.random() * 10, "source": "synthetic"},
        "temperature": {"temperature_c": 25 + random.random() * 10, "source": "synthetic"},
    }

"""
Configuration for South Sudan Cattle Movement Prediction System.
Central place for constants, paths, and region definitions.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BASELINE_DATA_DIR = DATA_DIR / "baseline"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# South Sudan bounding box (lat/lon)
# Source: Standard geographic bounds
SOUTH_SUDAN_BBOX = {
    "north": 12.2,
    "south": 3.5,
    "east": 35.9,
    "west": 23.9,
}

# Grid configuration: 10km x 10km cells
# At equator: 1 degree ≈ 111km, so 10km ≈ 0.09 degrees
# We use slightly finer resolution for South Sudan (lat 3.5-12.2)
GRID_CELL_SIZE_KM = 10
# Approximate degrees per 10km at mid-latitude (8°N)
DEG_PER_10KM_LAT = 0.09  # latitude: ~111km/degree
DEG_PER_10KM_LON = 0.10  # longitude: varies, ~100km/degree at 8°N

# Feature names for ML model
FEATURE_NAMES = [
    "ndvi",           # 0: Vegetation health (0-1)
    "rainfall",       # 1: Weekly rainfall sum (mm)
    "moisture",       # 2: Sentinel-1 VV backscatter
    "temperature",    # 3: Weekly average temp (°C)
    "previous_prob",  # 4: Last week's prediction
    "water_dist",     # 5: Distance to nearest water (km)
    "conflict",       # 6: Conflict within 50km (0/1)
]

NUM_FEATURES = len(FEATURE_NAMES)

# Google Earth Engine dataset IDs
GEE_DATASETS = {
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel1": "COPERNICUS/S1_GRD",
    "chirps": "UCSB-CHG/CHIRPS/DAILY",
    "era5": "ECMWF/ERA5/DAILY",
    "modis": "MODIS/006/MOD13Q1",
    "surface_water": "JRC/GSW1_4/GlobalSurfaceWater",
}

# FAO cattle density (for labels)
# Harvard Dataverse: doi:10.7910/DVN/GIVQ75
FAO_CATTLE_URL = "https://dataverse.harvard.edu/api/access/datafile/3403841"

# Cattle density threshold for binary labels (animals per km²)
# Adjust based on FAO raster scale
CATTLE_DENSITY_THRESHOLD = 13.0

# Season definitions (FEWS NET patterns)
DRY_SEASON_MONTHS = (11, 12, 1, 2, 3, 4)  # Nov-Apr
WET_SEASON_MONTHS = (5, 6, 7, 8, 9, 10)   # May-Oct

# Physical constraints
MAX_MOVEMENT_KM_PER_WEEK = 50
NDVI_TOO_DRY_THRESHOLD = 0.2
NDVI_LOW_PROBABILITY = 0.1

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BASELINE_DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

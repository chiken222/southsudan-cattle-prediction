"""
FAO Gridded Livestock of the World - Cattle density baseline.
Used for ground truth labels. Download and extract per-cell cattle density.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import json

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BASELINE_DATA_DIR, RAW_DATA_DIR


def download_fao_cattle_density(force: bool = False) -> Optional[Path]:
    """
    Download FAO cattle density GeoTIFF from Harvard Dataverse.
    Returns path to downloaded file, or None if failed.
    """
    # FAO GLW - Cattle 2015 Density
    # Direct link may change; user can manually download from:
    # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GIVQ75
    output_path = BASELINE_DATA_DIR / "Cattle_2015_Da.tif"

    if output_path.exists() and not force:
        return output_path

    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        return None

    # Harvard Dataverse API - get file URL
    # The persistent ID for FAO cattle is doi:10.7910/DVN/GIVQ75
    # File ID 3403841 is often used for Cattle_2015_Da.tif
    url = "https://dataverse.harvard.edu/api/access/datafile/3403841"
    # Note: This may require authentication for large files

    print("Downloading FAO cattle density (this may take a few minutes)...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from:")
        print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GIVQ75")
        print(f"  Save as: {output_path}")
        return None


def extract_cattle_density_per_cell(
    fao_tif_path: Path,
    cells: List[Dict],
) -> Dict[str, float]:
    """
    Extract mean cattle density for each grid cell from FAO GeoTIFF.
    Returns dict: cell_id -> density (animals per km²)
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from shapely.geometry import box
        import numpy as np
    except ImportError as e:
        print(f"Install rasterio, shapely: {e}")
        return {}

    if not fao_tif_path.exists():
        print(f"FAO file not found: {fao_tif_path}")
        return {}

    result = {}
    with rasterio.open(fao_tif_path) as src:
        for cell in cells:
            bbox = cell["bbox"]
            try:
                window = from_bounds(
                    bbox["west"], bbox["south"],
                    bbox["east"], bbox["north"],
                    src.transform,
                )
                data = src.read(1, window=window)
                # Mask nodata
                nodata = src.nodata or -9999
                valid = data[data != nodata]
                if len(valid) > 0:
                    mean_density = float(np.nanmean(valid))
                else:
                    mean_density = 0.0
            except Exception as e:
                mean_density = 0.0
            result[cell["cell_id"]] = max(0, mean_density)

    return result


def get_synthetic_cattle_density(cells: List[Dict]) -> Dict[str, float]:
    """
    Generate synthetic cattle density when FAO data unavailable.
    Based on known pastoral zones and environmental factors.
    """
    import math
    import random
    random.seed(42)  # Reproducible
    
    result = {}
    for cell in cells:
        lat, lon = cell["lat"], cell["lon"]
        
        # Pastoral zones (not a circle!)
        density = 0
        
        # Central grazing belt (7-10°N, 27-33°E)
        if 7 <= lat <= 10 and 27 <= lon <= 33:
            density += 15
        
        # Eastern grasslands (5-9°N, 32-35°E)
        if 5 <= lat <= 9 and 32 <= lon <= 35:
            density += 10
        
        # Western corridor (6-11°N, 25-28°E)
        if 6 <= lat <= 11 and 25 <= lon <= 28:
            density += 12
        
        # Add some noise for realism
        density += random.uniform(-3, 8)
        
        # Avoid extremes (deserts, swamps)
        if lat < 5 or lat > 11:  # Far south/north
            density *= 0.3
        
        result[cell["cell_id"]] = max(0.1, min(50, density))
    return result

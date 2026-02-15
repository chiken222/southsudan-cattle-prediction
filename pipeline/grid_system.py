"""
South Sudan 10km x 10km grid system.
Creates a regular grid of cells covering the country's bounding box.
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SOUTH_SUDAN_BBOX,
    DEG_PER_10KM_LAT,
    DEG_PER_10KM_LON,
    PROCESSED_DATA_DIR,
)


def create_grid_cells() -> List[Dict[str, Any]]:
    """
    Create 10km x 10km grid cells covering South Sudan.
    Returns list of cell dicts with cell_id, lat, lon, bbox, geometry.
    """
    north = SOUTH_SUDAN_BBOX["north"]
    south = SOUTH_SUDAN_BBOX["south"]
    east = SOUTH_SUDAN_BBOX["east"]
    west = SOUTH_SUDAN_BBOX["west"]

    cells = []
    cell_id = 0

    # Iterate from south to north, west to east
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            # Cell center
            center_lat = lat + DEG_PER_10KM_LAT / 2
            center_lon = lon + DEG_PER_10KM_LON / 2

            # Bounding box of cell
            cell_bbox = {
                "west": lon,
                "east": lon + DEG_PER_10KM_LON,
                "south": lat,
                "north": lat + DEG_PER_10KM_LAT,
            }

            # GeoJSON polygon (cell as rectangle)
            # GeoJSON coords: [lon, lat], closed ring
            coords = [
                [lon, lat],
                [lon + DEG_PER_10KM_LON, lat],
                [lon + DEG_PER_10KM_LON, lat + DEG_PER_10KM_LAT],
                [lon, lat + DEG_PER_10KM_LAT],
                [lon, lat],  # close ring
            ]

            cells.append({
                "cell_id": f"cell_{cell_id}",
                "cell_idx": cell_id,
                "lat": center_lat,
                "lon": center_lon,
                "bbox": cell_bbox,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })
            cell_id += 1
            lon += DEG_PER_10KM_LON
        lat += DEG_PER_10KM_LAT

    return cells


def get_grid_dimensions() -> Tuple[int, int]:
    """Return (n_rows, n_cols) for the grid."""
    north = SOUTH_SUDAN_BBOX["north"]
    south = SOUTH_SUDAN_BBOX["south"]
    east = SOUTH_SUDAN_BBOX["east"]
    west = SOUTH_SUDAN_BBOX["west"]

    height_deg = north - south
    width_deg = east - west

    n_rows = max(1, int(math.ceil(height_deg / DEG_PER_10KM_LAT)))
    n_cols = max(1, int(math.ceil(width_deg / DEG_PER_10KM_LON)))

    return n_rows, n_cols


def save_grid_metadata(cells: List[Dict[str, Any]], output_path: Path = None) -> Path:
    """Save grid cell metadata to JSON."""
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "grid_cells.json"

    n_rows, n_cols = get_grid_dimensions()
    metadata = {
        "n_cells": len(cells),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cell_size_km": 10,
        "bbox": SOUTH_SUDAN_BBOX,
        "cells": cells,
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


if __name__ == "__main__":
    cells = create_grid_cells()
    print(f"Created {len(cells)} grid cells")
    n_rows, n_cols = get_grid_dimensions()
    print(f"Grid dimensions: {n_rows} rows x {n_cols} cols")
    save_grid_metadata(cells)
    print(f"Saved to {PROCESSED_DATA_DIR / 'grid_cells.json'}")

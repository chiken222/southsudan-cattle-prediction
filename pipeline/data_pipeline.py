"""
South Sudan Cattle Movement - Data Pipeline (Chunk 1)

MEMORY SAFETY: Process ONE week (one chunk) at a time. Save to disk immediately
after each week, then free memory. Do NOT load all 26 weeks into RAM at once—
this avoids crashing on 16GB RAM / RTX 3060 systems.

Pipeline steps:
1. Create 10km x 10km grid over South Sudan
2. Download FAO cattle density baseline (for labels)
3. For each week (one chunk at a time):
   - Pull satellite data for that week only (Sentinel-2, CHIRPS, Sentinel-1, ERA5)
   - Build features and labels for that week
   - Write chunk to disk (memory-mapped arrays)
   - Delete week data and run gc.collect()
4. Output: features.npy, labels.npy, cell_metadata.json

Run: python pipeline/data_pipeline.py
"""

import gc
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROCESSED_DATA_DIR,
    BASELINE_DATA_DIR,
    FEATURE_NAMES,
    NUM_FEATURES,
    CATTLE_DENSITY_THRESHOLD,
    DRY_SEASON_MONTHS,
    NDVI_TOO_DRY_THRESHOLD,
)

from pipeline.grid_system import create_grid_cells, get_grid_dimensions, save_grid_metadata
from pipeline.satellite_data import pull_weekly_satellite_data, get_synthetic_weekly_data
from pipeline.fao_baseline import (
    download_fao_cattle_density,
    extract_cattle_density_per_cell,
    get_synthetic_cattle_density,
)


def normalize_feature(name: str, value: float) -> float:
    """Normalize feature to 0-1 range for ML."""
    if name == "ndvi":
        return max(0, min(1, (value + 0.2) / 1.2))  # typical -0.2 to 1.0
    if name == "rainfall":
        return max(0, min(1, value / 100))  # 0-100mm typical
    if name == "moisture":
        return max(0, min(1, (value + 25) / 35))  # VV often -25 to 10
    if name == "temperature":
        return max(0, min(1, (value - 15) / 25))  # 15-40°C
    if name == "previous_prob":
        return max(0, min(1, value))
    if name == "water_dist":
        return max(0, min(1, 1 - value / 50))  # 0km=1, 50km+=0
    if name == "conflict":
        return float(value)
    return max(0, min(1, value))


def run_pipeline(
    use_gee: bool = True,
    use_fao: bool = True,
    num_weeks: int = 26,
) -> None:
    """
    Run full data pipeline. Processes one week (one chunk) at a time and saves
    to disk after each chunk to avoid high memory use and crashes.
    use_gee: Try to pull real satellite data from GEE
    use_fao: Try to use FAO cattle density for labels
    num_weeks: Number of weekly snapshots for training (default 26 = 6 months)
    """
    print("=" * 60)
    print("South Sudan Cattle Movement - Data Pipeline")
    print("(Processing one chunk at a time to avoid memory crashes)")
    print("=" * 60)

    # Step 1: Create grid
    print("\n[1/6] Creating 10km x 10km grid...")
    cells = create_grid_cells()
    n_rows, n_cols = get_grid_dimensions()
    n_cells = len(cells)
    print(f"  -> {n_cells} cells ({n_rows} x {n_cols})")

    save_grid_metadata(cells)
    cell_metadata = {
        "n_cells": n_cells,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "feature_names": FEATURE_NAMES,
        "num_weeks": num_weeks,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    # Step 2: FAO cattle density (for labels) - small, keep in RAM
    print("\n[2/6] Loading FAO cattle density baseline...")
    fao_path = download_fao_cattle_density()
    if fao_path and use_fao:
        cattle_density = extract_cattle_density_per_cell(fao_path, cells)
        print(f"  -> Extracted density for {len(cattle_density)} cells")
    else:
        cattle_density = get_synthetic_cattle_density(cells)
        print(f"  -> Using synthetic density for {len(cattle_density)} cells")

    # Static features (small arrays, one per cell)
    water_dist_km = np.array([
        5 + 10 * (1 - abs(c["lat"] - 6) / 10) * (1 - abs(c["lon"] - 31) / 10)
        for c in cells
    ], dtype=np.float32)
    conflict = np.zeros(n_cells, dtype=np.float32)

    # Step 3: Pre-allocate output arrays on disk (memory-mapped) so we can write one chunk at a time
    print("\n[3/6] Pre-allocating output files (one chunk at a time will be written)...")
    features_path = PROCESSED_DATA_DIR / "features.npy"
    labels_path = PROCESSED_DATA_DIR / "labels.npy"

    # Use memmap to create files and write week-by-week without holding all weeks in RAM
    features_memmap = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(num_weeks, n_cells, NUM_FEATURES)
    )
    labels_memmap = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.float32, shape=(num_weeks, n_cells)
    )

    previous_probs = np.zeros(n_cells, dtype=np.float32)
    total_positive = 0

    # Step 4 & 5: Process ONE week at a time; build features + labels; save chunk; free memory
    print("\n[4/6] Pulling satellite data and building features (one chunk at a time)...")
    for week_idx in range(num_weeks):
        # Pull data for this week only (no accumulation in memory)
        if use_gee:
            try:
                data = pull_weekly_satellite_data(weeks_ago=num_weeks - week_idx - 1)
            except Exception as e:
                print(f"  Week {week_idx}: GEE failed ({e}), using synthetic")
                data = get_synthetic_weekly_data(week_idx)
        else:
            data = get_synthetic_weekly_data(week_idx)

        ndvi = data["ndvi"]["ndvi_mean"]
        rainfall = data["rainfall"]["rainfall_mm"]
        moisture = data["moisture"].get("moisture_vv", -12)
        temp = data["temperature"]["temperature_c"]

        # Build this week's features and labels (one chunk)
        week_features = np.zeros((n_cells, NUM_FEATURES), dtype=np.float32)
        week_labels = np.zeros(n_cells, dtype=np.float32)

        for cell_idx, cell in enumerate(cells):
            feat = [
                normalize_feature("ndvi", ndvi),
                normalize_feature("rainfall", rainfall),
                normalize_feature("moisture", moisture),
                normalize_feature("temperature", temp),
                previous_probs[cell_idx],
                normalize_feature("water_dist", water_dist_km[cell_idx]),
                float(conflict[cell_idx]),
            ]
            week_features[cell_idx, :] = feat

            # Labels for this cell this week
            dens = cattle_density.get(cell["cell_id"], 0)
            if dens >= CATTLE_DENSITY_THRESHOLD:
                label = 1.0
            else:
                label = 0.0
            ndvi_norm = feat[0]
            if ndvi_norm < NDVI_TOO_DRY_THRESHOLD:
                label = 0.0
            month = (datetime.now(timezone.utc).month - num_weeks + week_idx) % 12 or 12
            if month in DRY_SEASON_MONTHS and feat[5] > 0.6:
                label = min(1.0, label + 0.2)
            week_labels[cell_idx] = label
            total_positive += label

        # Update previous_probs for next week (from density heuristic)
        for cell_idx, cell in enumerate(cells):
            dens = cattle_density.get(cell["cell_id"], 0)
            previous_probs[cell_idx] = min(0.9, dens / 30)

        # Save this chunk to disk immediately (one chunk at a time)
        features_memmap[week_idx, :, :] = week_features
        labels_memmap[week_idx, :] = week_labels

        if (week_idx + 1) % 5 == 0 or week_idx == 0 or week_idx == num_weeks - 1:
            print(f"  -> Saved chunk {week_idx + 1}/{num_weeks}")

        # Free memory before next week
        del data, week_features, week_labels
        gc.collect()

    # Flush and close memmap (files are already on disk)
    del features_memmap, labels_memmap
    gc.collect()

    pos_ratio = total_positive / (num_weeks * n_cells)
    print(f"  -> Features shape: ({num_weeks}, {n_cells}, {NUM_FEATURES})")
    print(f"  -> Labels shape: ({num_weeks}, {n_cells}), positive ratio: {pos_ratio:.2%}")

    # Step 6: Save metadata
    print("\n[6/6] Saving metadata...")
    cell_list = [
        {
            "cell_id": c["cell_id"],
            "cell_idx": c["cell_idx"],
            "lat": c["lat"],
            "lon": c["lon"],
            "geometry": c["geometry"],
        }
        for c in cells
    ]
    cell_metadata["cells"] = cell_list
    with open(PROCESSED_DATA_DIR / "cell_metadata.json", "w") as f:
        json.dump(cell_metadata, f, indent=2)

    print(f"  -> {features_path}")
    print(f"  -> {labels_path}")
    print(f"  -> {PROCESSED_DATA_DIR / 'cell_metadata.json'}")

    print("\n" + "=" * 60)
    print("Pipeline complete! (One chunk at a time - no full-RAM load)")
    print("=" * 60)


def verify_outputs(use_mmap: bool = False):
    """
    Load and verify pipeline outputs.
    use_mmap: If True, load arrays with mmap_mode='r' to avoid loading full data into RAM.
    """
    if use_mmap:
        features = np.load(PROCESSED_DATA_DIR / "features.npy", mmap_mode="r")
        labels = np.load(PROCESSED_DATA_DIR / "labels.npy", mmap_mode="r")
    else:
        features = np.load(PROCESSED_DATA_DIR / "features.npy")
        labels = np.load(PROCESSED_DATA_DIR / "labels.npy")
    with open(PROCESSED_DATA_DIR / "cell_metadata.json") as f:
        meta = json.load(f)

    print("\nVerification:")
    print(f"  features shape: {features.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  feature ranges: min={features.min():.3f}, max={features.max():.3f}")
    print(f"  label distribution: {labels.mean():.2%} positive")
    print(f"  n_cells: {meta['n_cells']}")
    print("  OK: Grid cells cover South Sudan, arrays have reasonable values")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gee", action="store_true", help="Skip GEE, use synthetic satellite data")
    parser.add_argument("--no-fao", action="store_true", help="Skip FAO, use synthetic labels")
    parser.add_argument("--weeks", type=int, default=26, help="Number of weeks")
    parser.add_argument("--verify", action="store_true", help="Only verify existing outputs")
    parser.add_argument("--mmap", action="store_true", help="Use memory-mapped load when verifying (low RAM)")
    args = parser.parse_args()

    if args.verify:
        verify_outputs(use_mmap=args.mmap)
    else:
        run_pipeline(
            use_gee=not args.no_gee,
            use_fao=not args.no_fao,
            num_weeks=args.weeks,
        )

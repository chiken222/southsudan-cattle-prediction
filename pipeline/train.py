"""
Chunk 3: Model training for South Sudan cattle movement prediction.
- Loads processed features/labels from Chunk 1
- Train/val split, mixed precision (AMP), GPU memory safety
- Enforces NDVI threshold, recency penalty, optional movement constraint on pred
- Saves checkpoint every epoch and final model
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

# Project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.amp import autocast, GradScaler

from config import PROCESSED_DATA_DIR, MODELS_DIR
from models.architecture import create_model
from models.dataloader import create_dataloaders, get_grid_dims_from_meta
from models.gpu_utils import (
    BATCH_SIZE,
    get_device,
    print_gpu_memory,
    clear_gpu_cache,
    VRAM_SAFETY_LIMIT_GB,
)
from models.losses import CattleLossWithConstraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default training config (RTX 3060 safe)
SEQ_LEN = 4
TRAIN_RATIO = 0.8
MAX_EPOCHS = 50
LR = 1e-3
CLEAR_CACHE_EVERY_N_BATCHES = 10
PRINT_GPU_EVERY_N_BATCHES = 50
MAX_MOVEMENT_CELLS = 5  # ~50 km at 10 km/cell


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    use_amp,
    detect_anomaly,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (feat, lab) in enumerate(train_loader):
        feat = feat.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        lab_4d = lab.unsqueeze(1)
        # Recency: high previous_prob last 2 timesteps -> already grazed
        recency_mask = (feat[:, -1, 4] > 0.5) & (feat[:, -2, 4] > 0.5)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast("cuda"):
                logits = model(feat)
                loss = criterion(logits, lab_4d, ndvi=feat, recency_mask=recency_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if detect_anomaly:
                with torch.autograd.detect_anomaly():
                    logits = model(feat)
                    loss = criterion(logits, lab_4d, ndvi=feat, recency_mask=recency_mask)
                    loss.backward()
            else:
                logits = model(feat)
                loss = criterion(logits, lab_4d, ndvi=feat, recency_mask=recency_mask)
                loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % CLEAR_CACHE_EVERY_N_BATCHES == 0:
            clear_gpu_cache()
            gc.collect()
        if (batch_idx + 1) % PRINT_GPU_EVERY_N_BATCHES == 0 and device.type == "cuda":
            print_gpu_memory(f"epoch {epoch} batch {batch_idx + 1}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for feat, lab in val_loader:
        feat = feat.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        lab_4d = lab.unsqueeze(1)
        recency_mask = (feat[:, -1, 4] > 0.5) & (feat[:, -2, 4] > 0.5)
        logits = model(feat)
        loss = criterion(logits, lab_4d, ndvi=feat, recency_mask=recency_mask)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train cattle movement model (Chunk 3)")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR, help="Processed data directory")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR, help="Where to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max epochs")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default 4 for RTX 3060)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--detect-anomaly", action="store_true", help="Enable autograd anomaly detection (slow)")
    parser.add_argument("--checkpointing", action="store_true", help="Use gradient checkpointing to save VRAM")
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Device: %s", device)
    if device.type == "cuda":
        print_gpu_memory("before loading data")

    # Load data and create loaders
    train_loader, val_loader, meta = create_dataloaders(
        data_dir=args.data_dir,
        seq_len=SEQ_LEN,
        batch_size=args.batch_size,
        train_ratio=TRAIN_RATIO,
    )
    n_rows, n_cols = get_grid_dims_from_meta(meta)
    logger.info("Grid: %s x %s, train batches: %s", n_rows, n_cols, len(train_loader))

    # Model
    model = create_model(
        num_features=7,
        num_weeks=SEQ_LEN,
        grid_height=n_rows,
        grid_width=n_cols,
        use_checkpointing=args.checkpointing,
    ).to(device)
    criterion = CattleLossWithConstraints(
        bce_weight=1.0,
        ndvi_penalty_weight=2.0,
        recency_penalty_weight=0.3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda") if (device.type == "cuda" and not args.no_amp) else None
    use_amp = scaler is not None

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            use_amp=use_amp, detect_anomaly=args.detect_anomaly,
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info("Epoch %d  train_loss=%.4f  val_loss=%.4f", epoch, train_loss, val_loss)

        if device.type == "cuda":
            print_gpu_memory(f"after epoch {epoch}")

        # Checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "seq_len": SEQ_LEN,
        }
        torch.save(ckpt, args.out_dir / f"checkpoint_epoch_{epoch}.pth")
        logger.info("Saved checkpoint_epoch_%s.pth", epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out_dir / "cattle_model_best.pth")
            logger.info("Saved cattle_model_best.pth")

    # Final model (last epoch state)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "seq_len": SEQ_LEN,
        "num_features": 7,
    }, args.out_dir / "cattle_model.pth")
    logger.info("Saved cattle_model.pth (final)")

    if device.type == "cuda":
        alloc_gb = torch.cuda.memory_allocated() / 1024 ** 3
        if alloc_gb > VRAM_SAFETY_LIMIT_GB:
            logger.warning("GPU memory %.2f GB exceeded %.1f GB limit", alloc_gb, VRAM_SAFETY_LIMIT_GB)
        else:
            logger.info("GPU memory stayed under %.1f GB", VRAM_SAFETY_LIMIT_GB)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
cnn_train.py

Trains a multi-task Convolutional Neural Network (CNN) to perform both
colony counting (regression) and species classification. This script handles
the complete workflow from data loading and augmentation to model training,
validation, and checkpointing.
"""

# -------------------------
# Imports
# -------------------------
import os
import json
import time
import math
import random
import argparse
from pathlib import Path
from collections import OrderedDict, Counter
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------
# Configuration & Globals
# -------------------------
# Default settings for the training run. These can be overridden via command-line arguments.
DEFAULT_CFG = {
    # --- Paths ---
    "dataset_dir": "dataset",
    "training_lists_dir": "dataset/training_lists",
    "results_dir": "results_refactored",
    # --- Data & Loader ---
    "image_size": (224, 224),
    "batch_size": 32,
    "num_workers": 4,
    # --- Model Architecture ---
    "num_conv_blocks": 8,
    "base_filters": 32,
    "fc_size": 512,
    "dropout": 0.4,
    # --- Training Loop ---
    "epochs": 60,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # --- Loss & Metrics ---
    "class_loss_weight": 0.5, # Balances regression and classification losses.
    "label_smoothing": 0.05,
    # --- Checkpointing & Early Stopping ---
    "save_top_k": 3,
    "patience_es": 10,
    "min_delta": 1e-3,
    # --- Reproducibility ---
    "seed": 42,
}

# Master list of all possible classes in the dataset.
CLASSES = [
    "B.subtilis", "C.albicans", "Contamination", "Defect",
    "E.coli", "P.aeruginosa", "S.aureus"
]

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_id_list(path: str) -> List[str]:
    """Reads a list of image IDs from a text file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# -------------------------
# Data Handling
# -------------------------
class AgarMultiTaskDataset(Dataset):
    """
    Custom PyTorch Dataset for loading agar plate images and their associated
    colony count and species type. Handles missing files and JSON errors gracefully.
    """
    def __init__(self, txt_files: List[str], root_dir: str, class_map: Dict[str, int], transform: A.Compose = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.class_map = class_map

        # Aggregate unique image IDs from all provided list files, preserving order.
        all_ids = []
        for file_path in txt_files:
            all_ids.extend(read_id_list(file_path))
        self.ids = list(OrderedDict.fromkeys(all_ids))

        # Pre-generate paths for faster access during training.
        self.image_paths = [self.root / "images" / f"{i}.jpg" for i in self.ids]
        self.ann_paths = [self.root / "annotations" / f"{i}.json" for i in self.ids]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads a single data sample. Returns None if the sample is invalid,
        which is handled by the custom collate function.
        """
        try:
            # Load image
            img_path = str(self.image_paths[idx])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img_path}")

            # Load and parse annotation JSON
            with open(self.ann_paths[idx], 'r') as f:
                ann = json.load(f)

            # Extract regression target (count)
            count = float(ann.get("colonies_number", 0.0))

            # Extract classification target (species type)
            ctype = "Defect" # Default for empty or unlabeled plates
            if ann.get("labels") and len(ann["labels"]) > 0:
                ctype = ann['labels'][0]['class']

            class_idx = self.class_map.get(ctype)
            if class_idx is None: # Skip if class is not in our master list
                return None

            # Apply augmentations
            if self.transform:
                img = self.transform(image=img)['image']

            return {
                "image": img,
                "count": torch.tensor([count], dtype=torch.float32),
                "class": torch.tensor(class_idx, dtype=torch.long)
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # print(f"Warning: Skipping sample {self.ids[idx]} due to error: {e}")
            return None # The collate_fn will filter this out

def collate_filter_none(batch: List[Dict]) -> Dict:
    """
    A custom collate function for DataLoader that filters out `None` values.
    This is necessary because the Dataset returns `None` for corrupt samples.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "images": torch.stack([b['image'] for b in batch]),
        "counts": torch.stack([b['count'] for b in batch]),
        "classes": torch.stack([b['class'] for b in batch])
    }

# -------------------------
# Model Architecture
# -------------------------
class VanillaDeepCNN(nn.Module):
    """
    A custom multi-task CNN with a shared feature extraction body and two
    separate heads for regression (counting) and classification.
    """
    def __init__(self, in_channels: int = 3, num_conv_blocks: int = 8, base_filters: int = 32,
                 fc_size: int = 512, dropout: float = 0.4, num_classes: int = 7):
        super().__init__()

        # --- Shared Feature Extractor Body ---
        layers = []
        in_ch = in_channels
        for i in range(num_conv_blocks):
            out_ch = min(base_filters * (2**i), 512) # Double filters at each block, capped at 512
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_ch, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # --- Task-Specific Heads ---
        # 1. Regression head for colony counting
        self.head_count = nn.Sequential(
            nn.Linear(fc_size, fc_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_size // 2, 1)
        )
        # 2. Classification head for species type
        self.head_class = nn.Sequential(
            nn.Linear(fc_size, fc_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        shared_features = self.shared_fc(x)
        # Return outputs from both heads
        return self.head_count(shared_features), self.head_class(shared_features)

# -------------------------
# Training & Validation
# -------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    loss_count_fn: nn.Module, loss_class_fn: nn.Module, device: str,
                    scaler: torch.cuda.amp.GradScaler, scheduler: optim.lr_scheduler._LRScheduler,
                    epoch: int, writer: SummaryWriter, cfg: Dict):
    """Runs a single epoch of model training."""
    model.train()
    total_loss, n_samples = 0.0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}")

    for batch in pbar:
        if batch is None: continue
        images, counts, classes = batch["images"].to(device), batch["counts"].to(device), batch["classes"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision for performance
        with torch.cuda.amp.autocast(device_type=device, enabled=(scaler is not None)):
            pred_count, pred_class_logits = model(images)
            loss_c = loss_count_fn(pred_count, counts)
            loss_cl = loss_class_fn(pred_class_logits, classes)
            # Combine losses with a weighting factor
            loss = loss_c + cfg["class_loss_weight"] * loss_cl

        # Backpropagation
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step() # OneCycleLR updates every batch

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        pbar.set_postfix({"loss": f"{total_loss / n_samples:.4f}"})

    avg_loss = total_loss / n_samples if n_samples else 0.0
    if writer: writer.add_scalar("train/loss", avg_loss, epoch)
    return avg_loss

def validate_epoch(model: nn.Module, loader: DataLoader, device: str,
                   epoch: int, writer: SummaryWriter, class_list: List[str]) -> Tuple[float, float, str]:
    """Runs a single epoch of validation and returns key metrics."""
    model.eval()
    y_true_counts, y_pred_counts = [], []
    y_true_cls, y_pred_cls = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val E{epoch}"):
            if batch is None: continue
            images, counts, classes = batch["images"].to(device), batch["counts"], batch["classes"]

            preds_count, preds_class_logits = model(images)

            y_true_counts.extend(counts.cpu().numpy().flatten())
            y_pred_counts.extend(preds_count.cpu().numpy().flatten())
            y_true_cls.extend(classes.cpu().numpy().flatten())
            y_pred_cls.extend(preds_class_logits.argmax(dim=1).cpu().numpy().flatten())

    # Calculate metrics
    mae = mean_absolute_error(y_true_counts, y_pred_counts)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    report = classification_report(
        y_true_cls, y_pred_cls, labels=list(range(len(class_list))),
        target_names=class_list, zero_division=0
    )

    if writer:
        writer.add_scalar("val/mae", mae, epoch)
        writer.add_scalar("val/accuracy", acc, epoch)

    return mae, acc, report

# -------------------------
# Main Execution
# -------------------------
def run_training(cfg: Dict[str, Any]):
    """Orchestrates the entire training pipeline."""
    # 1. Setup
    set_seed(cfg["seed"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(results_dir / "tb"))
    class_map = {name: i for i, name in enumerate(CLASSES)}
    num_classes = len(CLASSES)
    print(f"Starting run. Results will be saved to: {results_dir}")

    # 2. Data Preparation
    train_files = list(Path(cfg["training_lists_dir"]).glob("*_train.txt"))
    test_files = list(Path(cfg["training_lists_dir"]).glob("*_val.txt"))
    if not train_files or not test_files:
        raise FileNotFoundError("Training/validation list files (*_train.txt, *_val.txt) not found.")

    train_tf = A.Compose([
        A.Resize(*cfg["image_size"]), A.RandomRotate90(p=0.5), A.Rotate(limit=25, p=0.7),
        A.OneOf([A.RandomBrightnessContrast(p=0.8), A.CLAHE(p=0.4)], p=0.7),
        A.GaussianBlur(p=0.25), A.GaussNoise(p=0.15),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])
    test_tf = A.Compose([
        A.Resize(*cfg["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_ds = AgarMultiTaskDataset(train_files, cfg["dataset_dir"], class_map, transform=train_tf)
    test_ds = AgarMultiTaskDataset(test_files, cfg["dataset_dir"], class_map, transform=test_tf)
    print(f"Dataset sizes: Train={len(train_ds)}, Validation={len(test_ds)}")

    # Calculate class weights for handling data imbalance
    print("Calculating class weights for loss function...")
    train_labels = [item['class'].item() for item in tqdm(train_ds, "Reading labels") if item]
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    # A logarithmic heuristic to weigh less frequent classes more heavily
    class_weights = torch.tensor([
        math.log(0.1 * total / (label_counts.get(i, 1))) for i in range(num_classes)
    ], dtype=torch.float32)
    class_weights = torch.clamp(class_weights, min=1.0).to(cfg["device"])
    print("Computed Class Weights:", [round(w.item(), 2) for w in class_weights])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate_filter_none)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate_filter_none)

    # 3. Model & Training Setup
    model = VanillaDeepCNN(num_conv_blocks=cfg["num_conv_blocks"], base_filters=cfg["base_filters"], fc_size=cfg["fc_size"], dropout=cfg["dropout"], num_classes=num_classes).to(cfg["device"])
    # SmoothL1Loss is robust to outliers, good for regression tasks like counting
    loss_count = nn.SmoothL1Loss()
    loss_class = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], steps_per_epoch=len(train_loader), epochs=cfg["epochs"], pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    # 4. Main Training Loop
    best_mae = float('inf')
    best_ckpts = []
    history = []
    epochs_no_improve = 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_count, loss_class, cfg["device"], scaler, scheduler, epoch, writer, cfg)
        mae, acc, report = validate_epoch(model, test_loader, cfg["device"], epoch, writer, CLASSES)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val MAE: {mae:.4f} | Val Acc: {acc:.4f} | Time: {elapsed:.1f}s")
        print("Validation Classification Report:\n", report)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": mae, "val_cls_acc": acc, "time_s": elapsed})

        # Checkpointing and Early Stopping
        if mae + cfg["min_delta"] < best_mae:
            best_mae = mae
            epochs_no_improve = 0
            ckpt_path = results_dir / f"best_epoch_{epoch:03d}_mae_{best_mae:.4f}.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            print(f"✔ New best model saved to {ckpt_path}")

            # Keep only the top K checkpoints
            best_ckpts.append(ckpt_path)
            best_ckpts = sorted(best_ckpts, key=lambda p: float(p.stem.split("_")[-1]))
            if len(best_ckpts) > cfg["save_top_k"]:
                os.remove(best_ckpts.pop(0)) # Remove the worst of the best
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience_es"]:
                print(f"Stopping early. No improvement in MAE for {cfg['patience_es']} epochs.")
                break

    # 5. Post-training Actions
    if best_ckpts:
        final_model_path = results_dir / "final_best_model.pth"
        best_state = torch.load(best_ckpts[0], map_location="cpu")
        torch.save(best_state["model_state"], final_model_path)
        print(f"Final best model state saved to: {final_model_path}")

    metadata = {"config": cfg, "history": history, "class_list": CLASSES}
    with open(results_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-task CNN for colony counting and classification.")

    # --- Add arguments from the default config ---
    for key, value in DEFAULT_CFG.items():
        arg_type = type(value)
        if arg_type == bool:
            parser.add_argument(f"--{key}", type=lambda x: (str(x).lower() == 'true'), default=value, help=f"Default: {value}")
        elif isinstance(value, (list, tuple)):
             parser.add_argument(f"--{key}", nargs='+', type=type(value[0]) if value else str, default=value, help=f"Default: {value}")
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"Default: {value}")

    args = parser.parse_args()
    config = vars(args)

    print("Running with configuration:")
    print(json.dumps(config, indent=2))
    run_training(config)
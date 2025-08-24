#!/usr/bin/env python3
"""
cnn_train.py

This script trains a multi-task (counting + classification) custom CNN.
It incorporates a professional workflow including:
- A flexible configuration system via command-line arguments.
- Reproducibility with a set random seed.
- Robust data loading that handles missing files and parses complex JSON.
- A custom CNN with a shared body and two heads for the different tasks.
- Advanced training techniques like a modern LR scheduler (OneCycleLR),
  weighted loss for class imbalance, and mixed-precision training.
- Detailed validation, checkpointing, and logging with TensorBoard.
"""
import os
import json
import time
import math
import random
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.tensorboard import SummaryWriter

# -------------------------
# CONFIG (Default settings for the training run)
# -------------------------
DEFAULT_CFG = {
    "dataset_dir": "dataset",
    "training_lists_dir": "dataset/training_lists",
    "results_dir": "results_enhanced",
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 60,
    "lr": 1e-3,                      # Max learning rate for the OneCycleLR scheduler
    "weight_decay": 1e-5,
    "num_workers": 4,
    "seed": 42,
    "num_conv_blocks": 8,             # Depth of the custom CNN
    "base_filters": 32,              # Width of the first layer
    "fc_size": 512,                  # Size of the fully connected layers
    "dropout": 0.4,
    "class_loss_weight": 0.5,        # Balances the two tasks. >0.5 prioritizes classification.
    "patience_es": 10,                 # Epochs to wait for improvement before early stopping
    "min_delta": 1e-3,                 # Minimum change in MAE to be considered an improvement
    "label_smoothing": 0.05,         # Helps prevent overconfidence in the classifier
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_top_k": 3,                   # Number of best model checkpoints to keep
}

# The master list of all possible classes in the dataset.
CLASSES = [
    "B.subtilis", "C.albicans", "Contamination", "Defect",
    "E.coli", "P.aeruginosa", "S.aureus"
]

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    """Ensures reproducibility by setting random seeds for all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_id_list(path):
    """Reads a list of image IDs from a text file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# -------------------------
# Dataset
# -------------------------
class AgarMultiTaskDataset(Dataset):
    """
    Custom Dataset for loading images and both of their labels (count and type).
    It robustly handles JSON parsing and cases where an image might be missing.
    """
    def __init__(self, txt_files, root_dir, class_map, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.class_map = class_map
        # Read all IDs from the provided text files
        self.ids = []
        for t in txt_files:
            self.ids.extend(read_id_list(t))
        # Remove any duplicate IDs while preserving the original order
        self.ids = list(OrderedDict.fromkeys(self.ids))
        # Pre-generate paths to images and annotations for faster access
        self.image_paths = [self.root / "images" / f"{i}.jpg" for i in self.ids]
        self.ann_paths = [self.root / "annotations" / f"{i}.json" for i in self.ids]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ids)

    def __getitem__(self, idx):
        """Loads and returns a single sample from the dataset at a given index."""
        imgp = str(self.image_paths[idx])
        annp = str(self.ann_paths[idx])
        try:
            # Load the image using OpenCV
            img = cv2.imread(imgp)
            if img is None: raise FileNotFoundError(f"Image not found {imgp}")

            # Load the corresponding JSON annotation
            with open(annp, 'r') as f:
                ann = json.load(f)

            # Get the regression target (the count)
            count = float(ann.get("colonies_number", 0.0))
            
            # Get the classification target (the type), handling empty plates
            if ann.get("labels") and len(ann["labels"]) > 0:
                ctype = ann['labels'][0]['class']
            else:
                ctype = "Defect" # Default class for empty plates
            
            # Convert the class name string to an integer index
            cidx = self.class_map.get(ctype, -1)
            if cidx == -1: return None # Skip if the class is not in our defined list

            # Apply data augmentation if a transform pipeline is provided
            if self.transform:
                img_t = self.transform(image=img)['image']
            else:
                img_t = ToTensorV2()(image=img)['image']

            # Return a dictionary of tensors
            return {"image": img_t, "count": torch.tensor([count]), "class": torch.tensor(cidx, dtype=torch.long)}
        except Exception:
            # If any error occurs (e.g., file not found, corrupt JSON), return None
            return None

def collate_filter_none(batch):
    """
    Custom function for the DataLoader that filters out any samples that
    failed to load (returned None) before stacking them into a batch.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return {
        "images": torch.stack([b['image'] for b in batch]),
        "counts": torch.stack([b['count'] for b in batch]),
        "classes": torch.stack([b['class'] for b in batch])
    }

# -------------------------
# Model
# -------------------------
class VanillaDeepCNN(nn.Module):
    """
    A custom multi-task CNN. It has a shared feature extraction "body"
    and two separate "heads" for the two different tasks.
    """
    def __init__(self, in_channels=3, num_conv_blocks=8, base_filters=32, fc_size=512, dropout=0.4, num_classes=7):
        super().__init__()
        # --- Shared Feature Extractor ("Body") ---
        # A series of convolutional blocks that learn to extract visual features
        layers, c, f = [], in_channels, base_filters
        for i in range(num_conv_blocks):
            out_ch = min(f * (2**i), 512) # Double filters each block, capped at 512
            layers += [nn.Conv2d(c, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True), nn.MaxPool2d(2)]
            c = out_ch
        self.features = nn.Sequential(*layers)
        
        # Global Average Pooling reduces each feature map to a single value
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # A shared fully-connected layer to process the features before the heads
        self.shared_fc = nn.Sequential(nn.Linear(c, fc_size), nn.ReLU(True), nn.Dropout(dropout))
        
        # --- Two Specialized "Heads" ---
        # Head 1: Predicts a single number for the colony count (Regression)
        self.head_count = nn.Sequential(nn.Linear(fc_size, fc_size//2), nn.ReLU(True), nn.Dropout(dropout/2), nn.Linear(fc_size//2, 1))
        # Head 2: Predicts a probability for each class (Classification)
        self.head_class = nn.Sequential(nn.Linear(fc_size, fc_size//2), nn.ReLU(True), nn.Dropout(dropout/2), nn.Linear(fc_size//2, num_classes))

    def forward(self, x):
        # The forward pass defines how data flows through the model
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # Flatten the features
        shared = self.shared_fc(x)
        # Return the output from both heads
        return self.head_count(shared), self.head_class(shared)

# -------------------------
# Training & Validation Loops
# -------------------------
def train_one_epoch(model, loader, optimizer, loss_count_fn, loss_class_fn, device, scaler, epoch, writer, cfg, scheduler):
    """Runs a single epoch of training."""
    model.train() # Set model to training mode
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    for batch in pbar:
        if batch is None: continue
        imgs, counts, classes = batch["images"].to(device), batch["counts"].to(device), batch["classes"].to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # Use Automatic Mixed Precision (AMP) for faster training on compatible GPUs
        with torch.amp.autocast(device_type=device, enabled=(scaler is not None)):
            pred_count, pred_class_logits = model(imgs)
            loss_c = loss_count_fn(pred_count, counts)
            loss_cl = loss_class_fn(pred_class_logits, classes)
            # Combine the two losses. The weight balances their importance.
            loss = loss_c + cfg["class_loss_weight"] * loss_cl
        
        if scaler: # If using AMP
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else: # Standard training
            loss.backward(); optimizer.step()
        
        if scheduler: scheduler.step() # Update the OneCycleLR scheduler every batch

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix({"loss": total_loss / (n + 1e-9)})
        
    avg_loss = total_loss / n if n else 0.0
    if writer: writer.add_scalar("train/loss", avg_loss, epoch) # Log to TensorBoard
    return avg_loss

def validate_epoch(model, loader, device, epoch, writer, class_list):
    """Runs a single epoch of validation."""
    model.eval() # Set model to evaluation mode
    y_true_counts, y_pred_counts, y_true_cls, y_pred_cls = [], [], [], []
    with torch.no_grad(): # Disable gradient calculations for speed
        for batch in tqdm(loader, desc=f"Val E{epoch}"):
            if batch is None: continue
            imgs, counts, classes = batch["images"].to(device), batch["counts"], batch["classes"]
            preds_count, preds_class_logits = model(imgs)
            
            # Store results for later calculation
            y_true_counts.extend(counts.cpu().numpy().flatten().tolist())
            y_pred_counts.extend(preds_count.cpu().numpy().flatten().tolist())
            y_true_cls.extend(classes.cpu().numpy().flatten().tolist())
            y_pred_cls.extend(preds_class_logits.argmax(dim=1).cpu().numpy().flatten().tolist())
            
    # Calculate all metrics
    mae = mean_absolute_error(y_true_counts, y_pred_counts)
    acc = accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0.0
    num_classes = len(class_list)
    class_report = classification_report(y_true_cls, y_pred_cls, labels=list(range(num_classes)), target_names=class_list, zero_division=0)
    
    if writer: # Log key metrics to TensorBoard
        writer.add_scalar("val/mae", mae, epoch)
        writer.add_scalar("val/accuracy", acc, epoch)
    return mae, acc, class_report

# -------------------------
# Main Runner
# -------------------------
def run_training(cfg):
    """The main function that orchestrates the entire training process."""
    set_seed(cfg["seed"])
    os.makedirs(cfg["results_dir"], exist_ok=True)
    tb = SummaryWriter(os.path.join(cfg["results_dir"], "tb"))
    class_list = CLASSES
    class_map = {c: i for i, c in enumerate(class_list)}
    num_classes = len(class_list)
    print("Class list:", class_list)

    # Find the data lists to use for training and validation
    train_files = [os.path.join(cfg["training_lists_dir"], f) for f in os.listdir(cfg["training_lists_dir"]) if f.endswith("_train.txt")]
    test_files = [os.path.join(cfg["training_lists_dir"], f) for f in os.listdir(cfg["training_lists_dir"]) if f.endswith("_val.txt")]
    if not train_files or not test_files:
        raise RuntimeError("Could not find _train.txt and/or _val.txt files. Please run a cleaning script first.")

    # Define the data augmentation pipelines
    train_tf = A.Compose([
        A.Resize(*cfg["image_size"]), A.RandomRotate90(p=0.5), A.Rotate(limit=25, p=0.7),
        A.OneOf([A.RandomBrightnessContrast(p=0.8), A.CLAHE(p=0.4)], p=0.7),
        A.GaussianBlur(p=0.25), A.GaussNoise(p=0.15),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
    ])
    test_tf = A.Compose([A.Resize(*cfg["image_size"]), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])

    # Create the Dataset and DataLoader instances
    train_ds = AgarMultiTaskDataset(train_files, cfg["dataset_dir"], class_map, transform=train_tf)
    test_ds = AgarMultiTaskDataset(test_files, cfg["dataset_dir"], class_map, transform=test_tf)
    print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    # Calculate weights for the classification loss to handle imbalanced data
    print("Calculating class weights...")
    train_labels = [item['class'].item() for item in tqdm(train_ds, desc="Reading labels") if item]
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor([math.log(0.1 * total / (label_counts.get(i, 1))) for i in range(num_classes)], dtype=torch.float32)
    class_weights = torch.clamp(class_weights, min=1.0).to(cfg["device"])
    print("Class weights:", [round(w.item(), 2) for w in class_weights])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate_filter_none)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate_filter_none)

    # Initialize the model, losses, optimizer, and scheduler
    model = VanillaDeepCNN(in_channels=3, num_conv_blocks=cfg["num_conv_blocks"], base_filters=cfg["base_filters"], fc_size=cfg["fc_size"], dropout=cfg["dropout"], num_classes=num_classes)
    device = cfg["device"]
    model.to(device)
    loss_count = nn.SmoothL1Loss()
    loss_class = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], steps_per_epoch=len(train_loader), epochs=cfg["epochs"], pct_start=0.1)
    scaler = torch.amp.GradScaler() if device.startswith("cuda") else None

    # Main training loop
    best_mae = float('inf'); best_ckpts = []; history = []; epochs_no_improve = 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_count, loss_class, device, scaler, epoch, tb, cfg, scheduler)
        mae, acc_cls, class_report = validate_epoch(model, test_loader, device, epoch, tb, class_list)
        elapsed = time.time() - t0
        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_mae: {mae:.4f} | cls_acc: {acc_cls:.4f} | time: {elapsed:.1f}s")
        print("Class report (val):\n", class_report)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": mae, "val_cls_acc": acc_cls, "time_s": elapsed})
        
        # Checkpointing logic: save the model if validation MAE improves
        if mae + cfg["min_delta"] < best_mae:
            best_mae = mae; epochs_no_improve = 0
            ckpt_path = os.path.join(cfg["results_dir"], f"best_epoch_{epoch:03d}_mae_{best_mae:.4f}.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            best_ckpts.append(ckpt_path)
            best_ckpts = sorted(best_ckpts, key=lambda p: float(Path(p).stem.split("_")[-1]))[:cfg["save_top_k"]]
            for p in list(Path(cfg["results_dir"]).glob("best_epoch_*.pth")):
                if str(p) not in best_ckpts: os.remove(p)
            print("Saved checkpoint:", ckpt_path)
        else:
            epochs_no_improve += 1
        
        # Early stopping: if performance hasn't improved for a while, stop training
        if epochs_no_improve >= cfg["patience_es"]:
            print("Early stopping triggered."); break
            
    # Save the single best model as 'final_best.pth'
    if best_ckpts:
        final_model_path = os.path.join(cfg["results_dir"], "final_best.pth")
        st = torch.load(best_ckpts[0], map_location="cpu")
        torch.save(st["model_state"], final_model_path)
        print("Final best saved to:", final_model_path)
        
    # Save all metadata for this training run
    metadata = {"cfg": cfg, "history": history, "class_list": class_list}
    with open(os.path.join(cfg["results_dir"], "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    tb.close()

if __name__ == "__main__":
    # --- Command-Line Interface ---
    # This allows you to change training parameters from the command line without editing the code
    p = ArgumentParser()
    for key, value in DEFAULT_CFG.items():
        if isinstance(value, (list, tuple)):
             p.add_argument(f"--{key}", nargs='+', type=type(value[0]) if value else str, default=value)
        elif isinstance(value, bool):
             p.add_argument(f"--{key}", type=lambda x: (str(x).lower() == 'true'), default=value)
        else:
             p.add_argument(f"--{key}", type=type(value), default=value)
    args = p.parse_args()
    cfg_override = vars(args)
    
    print("Running with config:", json.dumps(cfg_override, indent=2, default=str))
    run_training(cfg_override)
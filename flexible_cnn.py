import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import optuna

# --- Configuration ---
DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
RESULTS_DIR = 'results'
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'final_best_cnn_model.pth')
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, 'best_params.json')
ERROR_ANALYSIS_PATH = os.path.join(RESULTS_DIR, 'error_analysis')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs') # For TensorBoard

NUM_EPOCHS = 30
IMAGE_SIZE = (128, 128)
EARLY_STOPPING_PATIENCE = 7
N_OPTUNA_TRIALS = 50
TOP_K_ERRORS = 20

# --- 1. Integrated Data Cleaning Function ---
def clean_dataset_lists():
    print("--- Step 1: Cleaning dataset lists ---")
    try:
        existing_image_ids = {f.split('.')[0] for f in os.listdir(os.path.join(DATASET_DIR, 'images')) if f.endswith('.jpg')}
        if not existing_image_ids:
            print(f"Error: No '.jpg' files found in '{os.path.join(DATASET_DIR, 'images')}'. Exiting.")
            exit()
    except FileNotFoundError:
        print(f"Error: Image directory '{os.path.join(DATASET_DIR, 'images')}' not found. Exiting.")
        exit()

    for list_filename in os.listdir(TRAINING_LISTS_DIR):
        if not list_filename.endswith('.txt') or list_filename.endswith('_cleaned.txt'):
            continue

        original_path = os.path.join(TRAINING_LISTS_DIR, list_filename)
        cleaned_path = original_path.replace('.txt', '_cleaned.txt')
        
        with open(original_path, 'r') as f:
            content = f.read().strip()
            original_ids = []
            if content.startswith('[') and content.endswith(']'):
                try: original_ids = json.loads(content)
                except json.JSONDecodeError: pass
            if not original_ids:
                f.seek(0)
                original_ids = [line.strip() for line in f if line.strip()]

        cleaned_ids = [pid for pid in original_ids if pid in existing_image_ids]
        
        print(f"File '{list_filename}': Found {len(original_ids)} IDs, {len(cleaned_ids)} have matching images.")
        
        with open(cleaned_path, 'w') as f:
            for pid in cleaned_ids:
                f.write(f"{pid}\n")
    print("--- Cleaning complete. Using '_cleaned.txt' files for training. ---\n")

# --- 2. Data Handling and Model Architecture (No changes) ---
class AgarDataset(Dataset):
    def __init__(self, txt_files, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_ids = []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                self.sample_ids.extend([line.strip() for line in f if line.strip()])
        self.image_paths = [os.path.join(self.root_dir, 'images', f'{id}.jpg') for id in self.sample_ids]

    def __len__(self): return len(self.sample_ids)
    def __getitem__(self, idx):
        try:
            sample_id = self.sample_ids[idx]
            img_path = self.image_paths[idx]
            ann_path = os.path.join(self.root_dir, 'annotations', f'{sample_id}.json')
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(f"Img not found {img_path}")
            with open(ann_path, 'r') as f:
                count = float(json.load(f).get('colonies_number', 0))
            if self.transform: image = self.transform(image=image)['image']
            return {'image': image, 'label': torch.tensor([count], dtype=torch.float32), 'path': img_path, 'id': sample_id}
        except Exception: return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'paths': [item['path'] for item in batch],
        'ids': [item['id'] for item in batch]
    }

class FlexibleCNN(nn.Module):
    def __init__(self, image_size, num_conv_blocks, base_filters, fc_size, dropout_rate):
        super().__init__()
        layers = []
        in_channels, current_size = 1, image_size[0]
        for i in range(num_conv_blocks):
            out_channels = base_filters * (2**i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)])
            in_channels, current_size = out_channels, current_size // 2
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flattened_size = self.features(torch.zeros(1, 1, *image_size)).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, fc_size), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate), nn.Linear(fc_size, 1))
    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))

# --- 3. Optuna Objective Function (Updated with TQDM and TensorBoard) ---
def objective(trial, train_files, val_files):
    arch_params = {
        "num_conv_blocks": trial.suggest_int("num_conv_blocks", 3, 5),
        "base_filters": trial.suggest_categorical("base_filters", [16, 32]),
        "fc_size": trial.suggest_categorical("fc_size", [256, 512, 1024]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6)}
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    loss_type = trial.suggest_categorical("loss_type", ["L1", "SmoothL1"])
    
    # Setup TensorBoard writer for this trial
    writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, f"trial_{trial.number}"))
    writer.add_text("Trial/Parameters", json.dumps(trial.params, indent=4))
    print(f"\n--- Starting Trial {trial.number}: {trial.params} ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transform = A.Compose([A.Resize(*IMAGE_SIZE), A.HorizontalFlip(p=0.5), A.Rotate(30, p=0.5), A.GaussianBlur(p=0.2), A.RandomBrightnessContrast(p=0.3), A.Normalize((0.5,), (0.5,)), ToTensorV2()])
    val_transform = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize((0.5,), (0.5,)), ToTensorV2()])
    
    train_loader = DataLoader(AgarDataset(train_files, DATASET_DIR, train_transform), batch_size, True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(AgarDataset(val_files, DATASET_DIR, val_transform), batch_size*2, False, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    
    model = FlexibleCNN(IMAGE_SIZE, **arch_params).to(device)
    loss_fn = {"L1": nn.L1Loss(), "SmoothL1": nn.SmoothL1Loss()}[loss_type]
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.2)
    
    best_mae = float('inf')
    epochs_no_improve = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            if batch is None: continue
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        total_mae = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                if batch is None: continue
                images, labels = batch['images'].to(device), batch['labels'].to(device)
                outputs = model(images)
                total_mae += torch.abs(outputs - labels).sum().item()
        
        avg_mae = total_mae / len(val_loader.dataset) if val_loader.dataset else 0
        avg_loss = train_loss / len(train_loader) if train_loader else 0
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('MAE/Validation', avg_mae, epoch)
        scheduler.step(avg_mae)

        if avg_mae < best_mae: best_mae = avg_mae; epochs_no_improve = 0
        else: epochs_no_improve += 1
            
        trial.report(avg_mae, epoch)
        if trial.should_prune(): writer.close(); raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE: break
    
    writer.add_hparams(trial.params, {'hparam/best_mae': best_mae})
    writer.close()
    return best_mae

# --- 4. Final Training and Error Analysis (Updated with TQDM) ---
def train_final_model(params, train_files, val_files):
    print("\n--- Training Final Model with Best Parameters ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch_params = {k: v for k, v in params.items() if k in ["num_conv_blocks", "base_filters", "fc_size", "dropout_rate"]}
    
    train_transform = A.Compose([A.Resize(*IMAGE_SIZE), A.HorizontalFlip(p=0.5), A.Rotate(30,p=0.5), A.GaussianBlur(p=0.2), A.RandomBrightnessContrast(p=0.3), A.Normalize((0.5,), (0.5,)), ToTensorV2()])
    full_train_dataset = ConcatDataset([AgarDataset(train_files, DATASET_DIR, train_transform), AgarDataset(val_files, DATASET_DIR, train_transform)])
    loader = DataLoader(full_train_dataset, params['batch_size'], True, num_workers=2, collate_fn=collate_fn)
    
    model = FlexibleCNN(IMAGE_SIZE, **arch_params).to(device)
    loss_fn = {"L1": nn.L1Loss(), "SmoothL1": nn.SmoothL1Loss()}[params['loss_type']]
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'])

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"Final Training Epoch {epoch+1}/{NUM_EPOCHS}"):
            if batch is None: continue
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"-> Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

def analyze_errors(params, val_files):
    print("\n--- Analyzing Errors of Final Model ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch_params = {k: v for k, v in params.items() if k in ["num_conv_blocks", "base_filters", "fc_size", "dropout_rate"]}
    model = FlexibleCNN(IMAGE_SIZE, **arch_params)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device).eval()

    val_transform = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize((0.5,), (0.5,)), ToTensorV2()])
    val_loader = DataLoader(AgarDataset(val_files, DATASET_DIR, val_transform), params['batch_size']*2, False, collate_fn=collate_fn)
    
    errors = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing Errors"):
            if batch is None: continue
            images, labels, paths = batch['images'].to(device), batch['labels'], batch['paths']
            preds = model(images).cpu()
            errors.extend([{'error': abs(p.item()-l.item()), 'path': path, 'true': l.item(), 'pred': p.item()} for p, l, path in zip(preds, labels, paths)])
                
    errors.sort(key=lambda x: x['error'], reverse=True)
    os.makedirs(ERROR_ANALYSIS_PATH, exist_ok=True)
    print(f"Saving top {TOP_K_ERRORS} worst predictions to {ERROR_ANALYSIS_PATH}")
    
    for i, item in enumerate(errors[:TOP_K_ERRORS]):
        img = cv2.imread(item['path'])
        img = cv2.resize(img, (256, 256))
        text = f"True: {item['true']:.1f}, Pred: {item['pred']:.1f}, Err: {item['error']:.1f}"
        cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(ERROR_ANALYSIS_PATH, f"worst_{i+1:02d}.jpg"), img)

# --- Main Execution Block ---
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Automatically clean dataset lists to sync with available images
    clean_dataset_lists()
    
    # Step 2: Define paths to the newly created cleaned files
    train_cleaned_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith('_train_cleaned.txt')]
    val_cleaned_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith(('_val_cleaned.txt', '_test_cleaned.txt'))]

    if not train_cleaned_files or not val_cleaned_files:
        print("Error: Cleaned training or validation files not found. Please ensure original files exist and cleaning was successful.")
        exit()

    # Step 3: Run Optuna Hyperparameter Search
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_cleaned_files, val_cleaned_files), n_trials=N_OPTUNA_TRIALS)

    # Step 4: Save Best Parameters
    best_params = study.best_trial.params
    with open(BEST_PARAMS_PATH, 'w') as f: json.dump(best_params, f, indent=4)
    print(f"\n\n--- OPTUNA SEARCH COMPLETE ---\nBest MAE: {study.best_value:.4f}\nBest parameters saved to {BEST_PARAMS_PATH}\n{best_params}")
    
    # Step 5: Train Final Model using the best found parameters
    train_final_model(best_params, train_cleaned_files, val_cleaned_files)
    
    # Step 6: Analyze Final Model's Errors
    analyze_errors(best_params, val_cleaned_files)      
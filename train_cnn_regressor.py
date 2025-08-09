import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import shutil

# --- Configuration ---
DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
MODEL_SAVE_PATH = 'best_multitask_cnn_model.pth'
ERROR_ANALYSIS_DIR = 'error_analysis_multitask_cnn'
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
EARLY_STOPPING_PATIENCE = 7
# ---------------------

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- NEW: Function to automatically discover all classes ---
def discover_classes(dataset_dir, lists_dir):
    print("Discovering classes...")
    class_names = set()
    all_sample_ids = []

    # Aggregate all sample IDs from all list files
    for filename in os.listdir(lists_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(lists_dir, filename), 'r') as f:
                content = f.read().strip()
                ids = []
                if content.startswith('[') and content.endswith(']'):
                    try: ids = json.loads(content)
                    except json.JSONDecodeError: pass
                else:
                    f.seek(0)
                    ids = [line.strip() for line in f if line.strip()]
                all_sample_ids.extend(ids)
    
    # Check the JSON for each sample ID to find the class
    for sample_id in set(all_sample_ids): # Use set to avoid redundant checks
        ann_path = os.path.join(dataset_dir, 'annotations', f'{sample_id}.json')
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
                class_name = annotation.get('classes', [None])[0]
                if class_name:
                    class_names.add(class_name)
    
    sorted_classes = sorted(list(class_names))
    print(f" Discovered {len(sorted_classes)} classes: {sorted_classes}")
    return sorted_classes

class AgarDataset(Dataset):
    def __init__(self, txt_files, root_dir, class_to_idx, transform=None): # Added class_to_idx
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx # Store the class map
        self.sample_ids, self.image_paths = [], []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    try: ids = json.loads(content)
                    except json.JSONDecodeError: ids = []
                    self.sample_ids.extend(ids)
                else:
                    f.seek(0)
                    self.sample_ids.extend([line.strip() for line in f if line.strip()])
        self.image_paths = [os.path.join(self.root_dir, 'images', f'{id}.jpg') for id in self.sample_ids]

    def __len__(self): return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = self.image_paths[idx]
        ann_path = os.path.join(self.root_dir, 'annotations', f'{sample_id}.json')
        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError: return None
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
            count = float(annotation.get('colonies_number', 0))
            class_name = annotation.get('classes', [None])[0]
            # Use the provided map to get the class index
            class_idx = self.class_to_idx.get(class_name, -1)
            if class_idx == -1: return None
        if self.transform: image = self.transform(image)
        return image, (torch.tensor([count], dtype=torch.float32), torch.tensor(class_idx, dtype=torch.long))

class MultiTaskCNN(nn.Module):
    def __init__(self, initial_image_size, num_classes):
        super(MultiTaskCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *initial_image_size)
            self.flattened_size = self.features(dummy_input).view(1, -1).size(1)
        self.fc_shared = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.regression_head = nn.Linear(512, 1)
        self.classification_head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x); x = x.view(-1, self.flattened_size); x = self.fc_shared(x)
        return self.regression_head(x), self.classification_head(x)

def main():
    # --- UPDATED: Automatically discover classes and create mapping ---
    classes = discover_classes(DATASET_DIR, TRAINING_LISTS_DIR)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith('_train.txt')]
    val_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith(('_val.txt', '_test.txt'))]
    
    # Pass the class_to_idx map to the datasets
    train_dataset = AgarDataset(txt_files=train_txt_files, root_dir=DATASET_DIR, class_to_idx=class_to_idx, transform=train_transform)
    val_dataset = AgarDataset(txt_files=val_txt_files, root_dir=DATASET_DIR, class_to_idx=class_to_idx, transform=val_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"📊 Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    model = MultiTaskCNN(initial_image_size=IMAGE_SIZE, num_classes=len(classes))
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    best_mae = float('inf')
    epochs_no_improve = 0
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)
    
    print("\n Starting training for Multi-Task CNN...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            if data is None: continue
            images, (counts, class_labels) = data
            optimizer.zero_grad()
            count_outputs, class_outputs = model(images)
            loss_reg = criterion_regression(count_outputs, counts)
            loss_cls = criterion_classification(class_outputs, class_labels)
            total_loss = loss_reg + loss_cls
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        total_mae, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                if data is None: continue
                images, (counts, class_labels) = data
                count_outputs, class_outputs = model(images)
                total_mae += torch.abs(count_outputs - counts).sum().item()
                _, predicted_classes = torch.max(class_outputs.data, 1)
                total_correct += (predicted_classes == class_labels).sum().item()
                total_samples += class_labels.size(0)
        
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        scheduler.step(avg_mae)
        
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Val MAE: {avg_mae:.2f}, Val Acc: {accuracy:.2f}%")

        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   -> New best model saved with MAE: {best_mae:.2f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print(f"--- Finished Training ---\n Best model saved to {MODEL_SAVE_PATH} with MAE: {best_mae:.2f}")

if __name__ == '__main__':
    main()
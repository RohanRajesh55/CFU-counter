import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import shutil

# --- Configuration ---
# This section includes more advanced configuration options for a robust training pipeline.

DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
MODEL_SAVE_PATH = 'model/best_model.pth' # Path to save the best performing model.
ERROR_ANALYSIS_DIR = 'error_analysis'    # Directory to save images the model struggled with.
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224) # Image size required by the pre-trained EfficientNet model.
EARLY_STOPPING_PATIENCE = 5 # Number of epochs to wait for improvement before stopping training.

# --- For faster testing, limit the number of samples ---
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200
# ---------------------

# --- Helper Functions and Classes ---

def collate_fn(batch):
    """Filters out None values from a batch, essential for datasets with potential missing files."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

class AgarDataset(Dataset):
    """
    Custom Dataset to load agar plate images and their colony counts.
    It's enhanced to pre-cache image paths for easier access during error analysis.
    """
    def __init__(self, txt_files, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_ids = []
        self.image_paths = [] # Store image paths directly for easy retrieval.
        
        # Load sample IDs from provided text files.
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    try:
                        ids = json.loads(content)
                        self.sample_ids.extend(ids)
                    except json.JSONDecodeError:
                        pass # Silently ignore parsing errors
                else:
                    f.seek(0)
                    self.sample_ids.extend([line.strip() for line in f if line.strip()])
        
        # Truncate dataset if a maximum number of samples is specified.
        if max_samples is not None and len(self.sample_ids) > max_samples:
            self.sample_ids = self.sample_ids[:max_samples]
        
        # Pre-generate the full paths to all image files.
        self.image_paths = [os.path.join(self.root_dir, 'images', f'{id}.jpg') for id in self.sample_ids]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """Loads and returns a single sample (image, count) at the given index."""
        sample_id = self.sample_ids[idx]
        img_path = self.image_paths[idx] # Use the pre-generated path.
        ann_path = os.path.join(self.root_dir, 'annotations', f'{sample_id}.json')
        
        try:
            # Open image and convert to grayscale.
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            return None # The collate_fn will handle this.
            
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
            count = float(annotation.get('colonies_number', 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([count], dtype=torch.float32)

def main():
    # --- Data Loading and Transforms with Augmentation ---
    # Data augmentation is applied ONLY to the training set. It artificially expands the dataset
    # by creating modified versions of images, making the model more robust to variations.
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),      # Randomly flip images horizontally.
        transforms.RandomRotation(15),          # Randomly rotate images by up to 15 degrees.
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly change brightness and contrast.
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))    # Normalize for a single-channel (grayscale) image.
    ])

    # The validation transform does NOT include augmentation. We want to evaluate the model
    # on unmodified images to get a consistent and unbiased measure of its performance.
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith('_train.txt')]
    val_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith(('_val.txt', '_test.txt'))]

    train_dataset = AgarDataset(txt_files=train_txt_files, root_dir=DATASET_DIR, transform=train_transform, max_samples=MAX_TRAIN_SAMPLES)
    val_dataset = AgarDataset(txt_files=val_txt_files, root_dir=DATASET_DIR, transform=val_transform, max_samples=MAX_VAL_SAMPLES)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"📊 Using {len(train_dataset)} training samples.")
    print(f"📊 Using {len(val_dataset)} validation samples.")
    
    # --- Model, Loss, Optimizer, and Scheduler ---
    # Use transfer learning with a pre-trained EfficientNet model. This leverages knowledge
    # learned from the massive ImageNet dataset, which is much more effective than training from scratch.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # **Adapt the model for our specific task**:
    # 1. Modify the first convolutional layer to accept 1-channel (grayscale) images
    #    instead of the 3-channel (RGB) images it was originally trained on.
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    
    # 2. Replace the final classifier layer. The original layer was for 1000 ImageNet classes.
    #    We replace it with a new linear layer that outputs a single value for our regression task.
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    criterion = nn.MSELoss() # Use Mean Squared Error for the regression loss.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler: Automatically reduces the learning rate when the validation metric
    # (MAE) has stopped improving. This helps the model fine-tune its weights.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # --- Training Loop with Early Stopping and Error Analysis ---
    best_mae = float('inf') # Initialize with infinity to ensure the first epoch's MAE is lower.
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)
    
    print("\n🚀 Starting advanced training...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set the model to training mode (enables dropout, etc.).
        running_loss = 0.0
        for data in train_loader:
            if data is None: continue
            images, counts = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, counts)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode (disables dropout, etc.).
        total_mae = 0.0
        epoch_errors = [] # To store error information for analysis.
        with torch.no_grad(): # Disable gradient calculations to save memory and speed up evaluation.
            for i, data in enumerate(val_loader):
                if data is None: continue
                images, counts = data
                outputs = model(images)
                
                # Calculate Mean Absolute Error (MAE), which is more interpretable than MSE for evaluation.
                # MAE is the average absolute difference between predicted and actual counts.
                mae_batch = torch.abs(outputs - counts)
                total_mae += mae_batch.sum().item()
                
                # Collect data for error analysis.
                for j in range(images.size(0)):
                    sample_idx = i * BATCH_SIZE + j
                    if sample_idx < len(val_dataset):
                        error = mae_batch[j].item()
                        img_path = val_dataset.image_paths[sample_idx]
                        epoch_errors.append({'path': img_path, 'error': error, 'pred': outputs[j].item(), 'actual': counts[j].item()})
        
        avg_mae = total_mae / len(val_dataset) if len(val_dataset) > 0 else 0
        scheduler.step(avg_mae) # Update the LR scheduler based on validation MAE.
        
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Validation MAE: {avg_mae:.2f} colonies")

        # --- Check for Improvement and Save Best Model ---
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 🎉 New best model saved with MAE: {best_mae:.2f}")
            epochs_no_improve = 0
            
            # --- Perform Error Analysis ---
            # Save the top 5 worst predictions from this new best epoch for manual inspection.
            epoch_errors.sort(key=lambda x: x['error'], reverse=True)
            for k, err_info in enumerate(epoch_errors[:5]):
                fname = os.path.basename(err_info['path'])
                new_fname = f"worst_{k+1}_pred_{err_info['pred']:.1f}_actual_{err_info['actual']:.0f}_{fname}"
                shutil.copy(err_info['path'], os.path.join(ERROR_ANALYSIS_DIR, new_fname))
        else:
            epochs_no_improve += 1
        
        # --- Early Stopping Check ---
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print(f"\n--- Finished Training ---\n✅ Best model saved to {MODEL_SAVE_PATH} with MAE: {best_mae:.2f}")

if __name__ == '__main__':
    main()
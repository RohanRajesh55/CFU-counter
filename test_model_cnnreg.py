import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np

# --- Re-define Model and Dataset Classes ---
# These must be defined exactly as in the training script to load the model and data correctly.

class ColonyCounterCNN(nn.Module):
    def __init__(self, initial_image_size=(50, 50)):
        super(ColonyCounterCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *initial_image_size)
            dummy_output = self.pool(self.conv2(self.pool(self.conv1(dummy_input))))
            self.flattened_size = dummy_output.view(1, -1).size(1)
        self.fc1 = nn.Linear(self.flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AgarDataset(Dataset):
    def __init__(self, txt_files, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_ids = []
        self.image_paths = []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    try:
                        ids = json.loads(content)
                        self.sample_ids.extend(ids)
                    except json.JSONDecodeError: pass
                else:
                    f.seek(0)
                    self.sample_ids.extend([line.strip() for line in f if line.strip()])
        
        # Keep a direct mapping to image paths for easy access
        self.image_paths = {sid: os.path.join(self.root_dir, 'images', f'{sid}.jpg') for sid in self.sample_ids}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = self.image_paths[sample_id]
        ann_path = os.path.join(self.root_dir, 'annotations', f'{sample_id}.json')
        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            return None
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
            count = float(annotation.get('colonies_number', 0))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([count], dtype=torch.float32), sample_id


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Configuration ---
MODEL_PATH = 'model/cnn_colony_counter.pth'
DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
IMAGE_SIZE = (50, 50)  # Must match the training image size
BATCH_SIZE = 8

def test_model():
    """Main function to load the model and run evaluation."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Using device: {device}")

    # --- Load Model ---
    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = ColonyCounterCNN(initial_image_size=IMAGE_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("✅ Model loaded successfully.")

    # --- Load Test Data ---
    # The transforms must be the same as the validation transforms during training.
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith('_val.txt')]
    
    if not test_txt_files:
        print("❌ No test files found in 'dataset/training_lists/'. Make sure files ending with '_val.txt' exist.")
        return

    test_dataset = AgarDataset(txt_files=test_txt_files, root_dir=DATASET_DIR, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"📊 Found {len(test_dataset)} samples for testing.")

    # --- Evaluation Loop ---
    all_preds = []
    all_actuals = []
    all_sample_ids = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for data in test_loader:
            if data is None:
                continue
            images, counts, sample_ids = data
            images, counts = images.to(device), counts.to(device)
            
            outputs = model(images)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_actuals.extend(counts.cpu().numpy().flatten())
            all_sample_ids.extend(sample_ids)

    if not all_actuals:
        print("❌ No data was processed. Cannot calculate metrics.")
        return
        
    # --- Calculate Metrics ---
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    mae = np.mean(np.abs(all_preds - all_actuals))
    mse = np.mean((all_preds - all_actuals)**2)
    rmse = np.sqrt(mse)

    print("\n--- 📈 Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} colonies")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} colonies")
    print("----------------------------\n")

    # --- Visualize Some Predictions ---
    print("--- 🖼️ Prediction Examples ---")
    num_examples = min(5, len(all_actuals))
    for i in random.sample(range(len(all_actuals)), num_examples):
        sample_id = all_sample_ids[i]
        pred_count = all_preds[i]
        actual_count = all_actuals[i]
        
        print(f"Image: {sample_id}.jpg")
        print(f"  -> Predicted Count: {pred_count:.1f}")
        print(f"  -> Actual Count:    {actual_count:.1f}")
        print("-" * 20)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
    else:
        test_model()
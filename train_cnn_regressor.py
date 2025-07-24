import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# --- Configuration ---
DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
MODEL_SAVE_PATH = 'model/cnn_colony_counter.pth'
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 8

# --- For faster testing, limit the number of samples ---
# Set to None to use all available samples.
MAX_TRAIN_SAMPLES = 1000  # Use only 1000 images for training
MAX_VAL_SAMPLES = 200     # Use only 200 images for validation
# ---------------------

# This function filtmodel/ers out None values, which can happen if an image is missing.
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- CNN Model Definition ---
class ColonyCounterCNN(nn.Module):
    def __init__(self, initial_image_size=(50, 50)):
        super(ColonyCounterCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        # Helper to calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *initial_image_size)
            dummy_output = self.pool(self.conv2(self.pool(self.conv1(dummy_input))))
            self.flattened_size = dummy_output.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) # Output a single value for regression
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size) # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Custom DataLoader ---
class AgarDataset(Dataset):
    def __init__(self, txt_files, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_ids = []

        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    try:
                        ids = json.loads(content)
                        self.sample_ids.extend(ids)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse list format in {txt_file}")
                else:
                    f.seek(0)
                    self.sample_ids.extend([line.strip() for line in f if line.strip()])
        
        if max_samples is not None and len(self.sample_ids) > max_samples:
            self.sample_ids = self.sample_ids[:max_samples]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.root_dir, 'images', f'{sample_id}.jpg')
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
            
        count_tensor = torch.tensor([count], dtype=torch.float32)
        return image, count_tensor

# --- Main Training Logic ---
def main():
    image_size = (50, 50)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith('_train.txt')]
    val_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith(('_val.txt', '_test.txt'))]

    train_dataset = AgarDataset(txt_files=train_txt_files, root_dir=DATASET_DIR, transform=transform, max_samples=MAX_TRAIN_SAMPLES)
    val_dataset = AgarDataset(txt_files=val_txt_files, root_dir=DATASET_DIR, transform=transform, max_samples=MAX_VAL_SAMPLES)
    
    # --- THIS IS THE FIX ---
    # Setting num_workers=0 disables multiprocessing and avoids the error.
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"📊 Using {len(train_dataset)} training samples.")
    print(f"📊 Using {len(val_dataset)} validation samples.")

    model = ColonyCounterCNN(initial_image_size=image_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🚀 Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            if data is None: 
                continue
            images, counts = data

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, counts)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if len(train_loader) > 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], No data loaded for this epoch.")

    print("--- Finished Training ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # This check is important for multiprocessing, even with num_workers=0, it's good practice.
    main()
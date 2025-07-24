import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from torchvision import models

# --- Helper Classes and Functions ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

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
                        pass
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

        return image, torch.tensor([count], dtype=torch.float32)

# --- Load model ---
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# --- Predict single image ---
def predict_single_image(model, image_path, transform):
    try:
        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).unsqueeze(0)
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"🛑 Error: Cannot identify image file. It may be corrupted or not found: '{image_path}'")
        return
    except Exception as e:
        print(f"🛑 Unexpected error loading '{image_path}': {e}")
        return

    with torch.no_grad():
        output = model(image_tensor)
        predicted_count = output.item()

    print(f"\n🖼️ Image: {os.path.basename(image_path)}")
    print(f"🔬 Predicted Colony Count: {predicted_count:.2f}")

# --- Batch test function ---
def test_final_model(model, transform):
    # --- Configuration ---
    DATASET_DIR = 'dataset'
    TRAINING_LISTS_DIR = 'dataset/training_lists'
    BATCH_SIZE = 16
    MAX_TEST_SAMPLES = 200
    ACCURACY_TOLERANCE = 10

    test_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR)
                      if f.endswith(('_val.txt', '_test.txt'))]

    if not test_txt_files:
        print(f"🛑 Error: No files ending in '_val.txt' or '_test.txt' found in '{TRAINING_LISTS_DIR}'")
        return

    test_dataset = AgarDataset(
        txt_files=test_txt_files,
        root_dir=DATASET_DIR,
        transform=transform,
        max_samples=MAX_TEST_SAMPLES
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"📊 Testing on {len(test_dataset)} samples...")
    if len(test_dataset) == 0:
        print("Dataset is empty. Please check your training list files.")
        return

    total_mae = 0.0
    correct_predictions = 0
    all_actuals = []
    all_predictions = []

    with torch.no_grad():
        for data in test_loader:
            if data is None:
                continue
            images, counts = data
            outputs = model(images)

            abs_error = torch.abs(outputs - counts)
            correct_predictions += (abs_error <= ACCURACY_TOLERANCE).sum().item()
            total_mae += abs_error.sum().item()

            all_actuals.extend(counts.squeeze().tolist())
            all_predictions.extend(outputs.squeeze().tolist())

    avg_mae = total_mae / len(test_dataset)
    accuracy = (correct_predictions / len(test_dataset)) * 100

    print("\n--- Final Test Results ---")
    print(f"📈 Mean Absolute Error (MAE): {avg_mae:.2f} colonies")
    print(f"🎯 Accuracy (within ±{ACCURACY_TOLERANCE} colonies): {accuracy:.2f}%")

    print("\n--- Actual vs Predicted Counts ---")
    for actual, predicted in zip(all_actuals, all_predictions):
        print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# --- Entry Point ---
if __name__ == '__main__':
    MODEL_PATH = 'model/best_model.pth'
    IMAGE_SIZE = (224, 224)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = load_model(MODEL_PATH)

    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        if os.path.isfile(path_arg):
            predict_single_image(model, path_arg, transform)
        else:
            print(f"🛑 '{path_arg}' is not a valid image file.")
    else:
        test_final_model(model, transform)

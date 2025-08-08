import os
import json
import torch
import torch.nn as nn
import cv2
import time
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
RESULTS_DIR = 'results'
DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'final_best_cnn_model.pth')
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, 'best_params.json')
IMAGE_SIZE = (128, 128)

# --- Define Categories for Classification Metrics ---
# You can change these thresholds to better fit your experiment's needs
COUNT_BINS = [0, 20, 70, float('inf')]
CLASS_NAMES = ['Low (0-20)', 'Medium (21-70)', 'High (71+)']


# --- Re-define Model and Dataset classes (must match training script) ---
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
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(f"Img not found {img_path}")
            
            ann_path = os.path.join(self.root_dir, 'annotations', f'{self.sample_ids[idx]}.json')
            with open(ann_path, 'r') as f:
                count = float(json.load(f).get('colonies_number', 0))

            if self.transform: image = self.transform(image=image)['image']
            return {'image': image, 'label': torch.tensor([count], dtype=torch.float32), 'path': img_path}
        except Exception: return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'paths': [item['path'] for item in batch]
    }

# Helper function to convert counts to categories
def get_count_category(count):
    for i in range(len(COUNT_BINS) - 1):
        if COUNT_BINS[i] <= count <= COUNT_BINS[i+1]:
            return i
    return len(COUNT_BINS) - 2

def predict_single_image(model, device, image_path, params):
    """Loads a single image, makes a prediction, and draws the result."""
    print(f"\n--- Predicting single image: {image_path} ---")
    transform = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize((0.5,), (0.5,)), ToTensorV2()])
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not read image at {image_path}"); return
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_tensor = transform(image=image_gray)['image'].unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        prediction = model(image_tensor)
    end_time = time.time()
    predicted_count = prediction.item()
    rounded_prediction = round(predicted_count)
    print(f"Predicted Colony Count: {rounded_prediction}")
    print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
    output_image = cv2.resize(image_bgr, (512, 512))
    text = f"Predicted Count: {rounded_prediction}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(output_image, (5, 5), (10 + text_width, 20 + text_height), (0, 0, 0), -1)
    cv2.putText(output_image, text, (10, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    output_folder = os.path.join(RESULTS_DIR, 'single_predictions')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"pred_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, output_image)
    print(f"Visual result saved to: {output_path}")

def evaluate_on_test_set(model, device, params):
    """Evaluates the model, prints metrics, and saves visual results."""
    print("\n--- Evaluating model on the full test/validation set ---")
    test_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR) if f.endswith(('_val_cleaned.txt', '_test_cleaned.txt'))]
    if not test_files:
        print("Error: No '_val_cleaned.txt' or '_test_cleaned.txt' files found."); return
    print("Found the following files for evaluation:")
    for f in test_files: print(f"  - {os.path.basename(f)}")
    transform = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize((0.5,), (0.5,)), ToTensorV2()])
    test_dataset = AgarDataset(test_files, DATASET_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=params.get('batch_size', 32) * 2, shuffle=False, num_workers=2, collate_fn=collate_fn)
    all_labels, all_preds, total_time, total_images = [], [], 0, 0
    output_folder = os.path.join(RESULTS_DIR, 'test_set_predictions')
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving visual results for the test set in: '{output_folder}'")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test/Validation Set"):
            if batch is None: continue
            images, labels, paths = batch['images'].to(device), batch['labels'], batch['paths']
            start_time = time.time()
            preds = model(images)
            end_time = time.time()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_time += (end_time - start_time)
            total_images += len(images)
            for i in range(len(paths)):
                img_path, true_count, pred_count = paths[i], labels[i].item(), preds[i].item()
                output_image = cv2.imread(img_path)
                output_image = cv2.resize(output_image, (512, 512))
                cv2.putText(output_image, f"True: {round(true_count)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(output_image, f"Pred: {round(pred_count)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
                cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), output_image)
    if total_images == 0:
        print("\nError: No valid data was loaded. Cannot calculate metrics."); return
    y_true, y_pred = np.array(all_labels).flatten(), np.array(all_preds).flatten()
    # --- Regression Metrics (Primary Evaluation) ---
    mae, rmse, r2 = mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)
    avg_time_ms = (total_time / total_images) * 1000
    print("\n--- Regression Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f} (Average count error)")
    print(f"Root Mean Squared Error:   {rmse:.4f} (Penalizes large errors)")
    print(f"R-squared (R2) Score:      {r2:.4f} (Closer to 1 is better)")
    print("---------------------------------------")
    print(f"Average Inference Time:    {avg_time_ms:.2f} ms per image")
    # --- Classification Metrics (Secondary View) ---
    y_true_class = np.array([get_count_category(c) for c in y_true])
    y_pred_class = np.array([get_count_category(c) for c in y_pred])
    print("\n--- Classification Performance Metrics ---")
    print("Metrics for predicting count categories (Low, Medium, High)\n")
    print(f"Overall Accuracy: {accuracy_score(y_true_class, y_pred_class):.4f}")
    print("\nDetailed Report per Class:")
    print(classification_report(y_true_class, y_pred_class, target_names=CLASS_NAMES, zero_division=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the trained colony counting CNN.")
    parser.add_argument('--image', type=str, help="Path to a single image for prediction.")
    args = parser.parse_args()
    try:
        with open(BEST_PARAMS_PATH, 'r') as f: best_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Cannot find '{BEST_PARAMS_PATH}'. Please run the training script first."); exit()
    print("--- Using Best Hyperparameters ---"); print(json.dumps(best_params, indent=4)); print("----------------------------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    arch_params = {k: v for k, v in best_params.items() if k in ["num_conv_blocks", "base_filters", "fc_size", "dropout_rate"]}
    model = FlexibleCNN(image_size=IMAGE_SIZE, **arch_params)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Cannot find model weights at '{MODEL_SAVE_PATH}'. Please run the training script first."); exit()
    model.to(device)
    model.eval()
    if args.image:
        predict_single_image(model, device, args.image, best_params)
    else:
        evaluate_on_test_set(model, device, best_params)
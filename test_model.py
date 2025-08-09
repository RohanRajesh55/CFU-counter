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
# These classes and functions are needed to load the data and model correctly.
# They must match the definitions used in your training script.

def collate_fn(batch):
    """
    A custom collate function to filter out None values from a batch.
    This prevents crashes if an image file is missing or corrupted and __getitem__ returns None.
    """
    # Remove any samples that are None
    batch = list(filter(lambda x: x is not None, batch))
    # If the whole batch was None, return None so the training loop can skip it
    if not batch:
        return None
    # If there's valid data, use the default PyTorch function to stack it into a batch
    return torch.utils.data.dataloader.default_collate(batch)

class AgarDataset(Dataset):
    """
    Custom Dataset class to load images and their corresponding colony counts.
    """
    def __init__(self, txt_files, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_ids = []

        # Read all the sample IDs from the provided text files
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

        # Limit the number of samples if max_samples is set (for faster testing)
        if max_samples is not None and len(self.sample_ids) > max_samples:
            self.sample_ids = self.sample_ids[:max_samples]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """Loads and returns one sample (image, count) from the dataset."""
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.root_dir, 'images', f'{sample_id}.jpg')
        ann_path = os.path.join(self.root_dir, 'annotations', f'{sample_id}.json')

        try:
            # Open the image file and convert to grayscale ("L")
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            # If the image is not found, return None so collate_fn can handle it
            return None

        # Load the annotation JSON to get the colony count
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
            # Default to 0 if the key is missing
            count = float(annotation.get('colonies_number', 0))

        # Apply transformations (e.g., resizing, normalizing) if they are defined
        if self.transform:
            image = self.transform(image)

        # Return the transformed image and the count as a tensor
        return image, torch.tensor([count], dtype=torch.float32)

# --- Load model ---
def load_model(model_path):
    """
    Builds the EfficientNet-B0 model architecture and loads the saved weights.
    """
    # 1. Create an empty "shell" of the model architecture. weights=None means don't load pre-trained ImageNet weights.
    model = models.efficientnet_b0(weights=None)
    
    # 2. Modify the architecture to match the one used during training (grayscale input, single regression output).
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    # 3. Load the saved weights from your .pth file into the model shell.
    model.load_state_dict(torch.load(model_path))
    
    # 4. Set the model to evaluation mode. This disables training-specific layers like Dropout. CRITICAL for testing.
    model.eval()
    return model

# --- Predict single image ---
def predict_single_image(model, image_path, transform):
    """
    Loads a single image, preprocesses it, and predicts the colony count.
    """
    try:
        # Open the image and convert to grayscale
        image = Image.open(image_path).convert("L")
        # Apply the transformations and add a "batch" dimension with .unsqueeze(0)
        # The model expects a 4D tensor: [batch_size, channels, height, width]
        image_tensor = transform(image).unsqueeze(0)
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"Error: Cannot identify image file. It may be corrupted or not found: '{image_path}'")
        return
    except Exception as e:
        print(f"Unexpected error loading '{image_path}': {e}")
        return

    # Disable gradient calculation for inference to save memory and speed up
    with torch.no_grad():
        # Get the model's prediction
        output = model(image_tensor)
        # Extract the single floating-point number from the output tensor
        predicted_count = output.item()

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted Colony Count: {predicted_count:.2f}")

# --- Batch test function ---
def test_final_model(model, transform):
    """
    Runs a full evaluation on the test dataset and prints performance metrics.
    """
    # --- Configuration specific to the test ---
    DATASET_DIR = 'dataset'
    TRAINING_LISTS_DIR = 'dataset/training_lists'
    BATCH_SIZE = 16
    MAX_TEST_SAMPLES = 200 # Limit samples for a quick test, set to None for full test
    ACCURACY_TOLERANCE = 10 # Define "correct" as a prediction within +/- 10 colonies of the actual count

    # Find all test and validation list files
    test_txt_files = [os.path.join(TRAINING_LISTS_DIR, f) for f in os.listdir(TRAINING_LISTS_DIR)
                      if f.endswith(('_val.txt', '_test.txt'))]

    if not test_txt_files:
        print(f"Error: No files ending in '_val.txt' or '_test.txt' found in '{TRAINING_LISTS_DIR}'")
        return

    # Create the dataset and dataloader using the test files
    test_dataset = AgarDataset(
        txt_files=test_txt_files,
        root_dir=DATASET_DIR,
        transform=transform, # Use the validation transform (no augmentation)
        max_samples=MAX_TEST_SAMPLES
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"Testing on {len(test_dataset)} samples...")
    if len(test_dataset) == 0:
        print("Dataset is empty. Please check your training list files.")
        return

    # Initialize variables to store results
    total_mae = 0.0
    correct_predictions = 0
    all_actuals = []
    all_predictions = []

    # The main evaluation loop
    with torch.no_grad(): # Disable gradients for speed
        for data in test_loader:
            if data is None: continue # Skip empty batches
            
            images, counts = data
            outputs = model(images)

            # Calculate metrics for this batch
            abs_error = torch.abs(outputs - counts)
            # Count how many predictions were within the tolerance range
            correct_predictions += (abs_error <= ACCURACY_TOLERANCE).sum().item()
            # Add the batch's total absolute error to the running total
            total_mae += abs_error.sum().item()

            # Store all predictions and actuals for later display
            all_actuals.extend(counts.squeeze().tolist())
            all_predictions.extend(outputs.squeeze().tolist())

    # --- Calculate and Print Final Metrics ---
    # Calculate average Mean Absolute Error
    avg_mae = total_mae / len(test_dataset)
    # Calculate accuracy percentage based on the tolerance
    accuracy = (correct_predictions / len(test_dataset)) * 100

    print("\n--- Final Test Results ---")
    print(f"Mean Absolute Error (MAE): {avg_mae:.2f} colonies")
    print(f"Accuracy (within ±{ACCURACY_TOLERANCE} colonies): {accuracy:.2f}%")

    # Optionally, print all individual results for detailed analysis
    print("\n--- Actual vs Predicted Counts ---")
    for actual, predicted in zip(all_actuals, all_predictions):
        print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# --- Entry Point ---
# This is the main part of the script that runs when you execute `python test_model.py`
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = 'model/best_model.pth'
    IMAGE_SIZE = (224, 224)

    # Define the image transformations. Must be the same as the validation transform during training.
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the trained model
    model = load_model(MODEL_PATH)

    # --- Dual-Mode Logic ---
    # Check if a command-line argument (e.g., an image path) was provided
    if len(sys.argv) > 1:
        # If yes, run in single-image prediction mode
        path_arg = sys.argv[1]
        if os.path.isfile(path_arg):
            predict_single_image(model, path_arg, transform)
        else:
            print(f"'{path_arg}' is not a valid image file.")
    else:
        # If no argument was provided, run in full batch testing mode
        test_final_model(model, transform)
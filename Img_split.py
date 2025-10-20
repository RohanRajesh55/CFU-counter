import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- 1. SET YOUR PATHS HERE ---

# Path to your CNN project's "dataset" folder
SOURCE_DATA_DIR = Path("dataset")

# Path to your YOLO project folder (the one in your YAML's 'path' field)
YOLO_PROJECT_DIR = Path("/Users/shreyas/Desktop/Shreyas_Rohan")

# Folder containing your _train.txt and _val.txt files
TRAINING_LISTS_DIR = SOURCE_DATA_DIR / "training_lists"

# Folder where ALL your source images are
SOURCE_IMAGES_DIR = SOURCE_DATA_DIR / "images"

# Folder where ALL your YOLO-formatted .txt label files are
# (You must have converted these from your JSONs already)
SOURCE_LABELS_DIR = SOURCE_DATA_DIR / "labels_yolo" # <-- Make sure this path is correct!

# --- 2. DEFINE DESTINATION FOLDERS ---
train_img_dest = YOLO_PROJECT_DIR / "images" / "train"
val_img_dest = YOLO_PROJECT_DIR / "images" / "val"
train_label_dest = YOLO_PROJECT_DIR / "labels" / "train"
val_label_dest = YOLO_PROJECT_DIR / "labels" / "val"

# Create destination folders if they don't exist
os.makedirs(train_img_dest, exist_ok=True)
os.makedirs(val_img_dest, exist_ok=True)
os.makedirs(train_label_dest, exist_ok=True)
os.makedirs(val_label_dest, exist_ok=True)

# --- 3. HELPER FUNCTION TO PROCESS A LIST ---
def process_list(list_file, img_dest, label_dest):
    """Reads a list file and copies images and labels to their destination."""
    print(f"--- Processing {list_file.name} ---")
    with open(list_file, 'r') as f:
        file_ids = [line.strip() for line in f if line.strip()]

    copied_count = 0
    for file_id in tqdm(file_ids, desc=f"Copying to {img_dest.parent.name}"):
        img_src = SOURCE_IMAGES_DIR / f"{file_id}.jpg"
        label_src = SOURCE_LABELS_DIR / f"{file_id}.txt"
        
        img_dst_path = img_dest / f"{file_id}.jpg"
        label_dst_path = label_dest / f"{file_id}.txt"

        # Copy files only if they exist
        if img_src.exists() and label_src.exists():
            shutil.copy(img_src, img_dst_path)
            shutil.copy(label_src, label_dst_path)
            copied_count += 1
        else:
            if not img_src.exists():
                print(f"Warning: Image file not found, skipping: {img_src}")
            if not label_src.exists():
                print(f"Warning: Label file not found, skipping: {label_src}")
                
    print(f"Finished. Copied {copied_count} image/label pairs.")

# --- 4. FIND AND PROCESS ALL TRAIN/VAL LISTS ---
train_lists = list(TRAINING_LISTS_DIR.glob("*_train.txt"))
val_lists = list(TRAINING_LISTS_DIR.glob("*_val.txt"))

if not train_lists and not val_lists:
    print(f"Error: No _train.txt or _val.txt files found in {TRAINING_LISTS_DIR}")
else:
    # Process all training lists
    for list_file in train_lists:
        process_list(list_file, train_img_dest, train_label_dest)
        
    # Process all validation lists
    for list_file in val_lists:
        process_list(list_file, val_img_dest, val_label_dest)

print("\n--- File splitting complete! ---")
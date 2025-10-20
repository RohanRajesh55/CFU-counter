import json
import os
from pathlib import Path
from tqdm import tqdm
import cv2

# --- 1. CONFIGURATION ---

# These MUST match the 'names' list in your YOLO .yaml file, in the exact same order.
CLASSES = [
    "B.subtilis",
    "C.albicans",
    "Contamination",
    "Defect",
    "E.coli",
    "P.aeruginosa",
    "S.aureus"
]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

# --- 2. SET YOUR PATHS ---

# Path to the folder containing your original JSON annotation files
SOURCE_ANNOTATION_DIR = Path("dataset/annotations") 

# Path to the folder containing your original .jpg images
# This is needed to get image dimensions if they aren't in the JSON.
SOURCE_IMAGE_DIR = Path("dataset/images") 

# Path to the new folder where YOLO .txt files will be saved
OUTPUT_LABEL_DIR = Path("dataset/labels_yolo") 

# --- 3. CREATE OUTPUT DIRECTORY ---
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# --- 4. CONVERSION HELPER FUNCTION ---
def convert_to_yolo(bbox, img_w, img_h):
    """
    Converts a [x_min, y_min, x_max, y_max] bounding box
    to YOLO's [x_center_norm, y_center_norm, w_norm, h_norm] format.
    """
    x_min, y_min, x_max, y_max = bbox
    
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return None
        
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    
    # Normalize
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    return (x_center_norm, y_center_norm, w_norm, h_norm)

# --- 5. MAIN CONVERSION LOOP ---
json_files = list(SOURCE_ANNOTATION_DIR.glob("*.json"))
print(f"Found {len(json_files)} JSON files in {SOURCE_ANNOTATION_DIR}")

for json_path in tqdm(json_files, desc="Converting to YOLO format"):
    file_id = json_path.stem
    output_txt_path = OUTPUT_LABEL_DIR / f"{file_id}.txt"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # --- Get Image Dimensions ---
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")

        if img_w is None or img_h is None:
            img_path = SOURCE_IMAGE_DIR / f"{file_id}.jpg"
            if not img_path.exists():
                print(f"Warning: Cannot find image {img_path} to get dimensions. Skipping {file_id}")
                continue
            img = cv2.imread(str(img_path))
            img_h, img_w, _ = img.shape
        
        if img_w == 0 or img_h == 0:
            print(f"Warning: Invalid image dimensions (0) for {file_id}. Skipping.")
            continue

        yolo_lines = []
        
        # --- Extract Bounding Boxes ---
        # This loop is now corrected to read x, y, width, and height.
        for label in data.get("labels", []):
            
            class_name = label.get("class") 
            class_id = CLASS_MAP.get(class_name)
            
            # Get individual coordinates from the JSON
            x_min = label.get("x")
            y_min = label.get("y")
            width = label.get("width")
            height = label.get("height")

            if class_id is None:
                print(f"Warning: Skipping unknown class '{class_name}' in {json_path.name}")
                continue
            
            # Check if all coordinate keys exist
            if x_min is None or y_min is None or width is None or height is None:
                print(f"Warning: Missing one or more coordinate keys (x, y, width, height) in {json_path.name}. Skipping label.")
                continue
                
            # Create the [xmin, ymin, xmax, ymax] bbox list
            # We assume 'x' and 'y' are the top-left corner
            x_max = x_min + width
            y_max = y_min + height
            bbox = [x_min, y_min, x_max, y_max]
            
            # Convert [xmin, ymin, xmax, ymax] to YOLO format
            yolo_bbox = convert_to_yolo(bbox, img_w, img_h)
            
            if yolo_bbox:
                x_norm, y_norm, w_norm, h_norm = yolo_bbox
                yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Write the .txt file for this image
        with open(output_txt_path, 'w') as f:
            f.write("\n".join(yolo_lines))

    except Exception as e:
        print(f"Error processing {json_path.name}: {e}")

print("\n--- Conversion complete! ---")
print(f"YOLO labels saved to: {OUTPUT_LABEL_DIR}")
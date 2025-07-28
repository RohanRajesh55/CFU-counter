import json
from PIL import Image, ImageDraw

# --- 1. DEFINE FILE PATHS ---
# Change these to the image and annotation you want to check
image_path = 'dataset/images/378.jpg'
annotation_path = 'dataset/annotations/378.json'

try:
    # --- 2. LOAD THE IMAGE AND ANNOTATION FILE ---
    image = Image.open(image_path)
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # --- 3. PREPARE TO DRAW ON THE IMAGE ---
    draw = ImageDraw.Draw(image)

    # --- 4. LOOP THROUGH EACH COLONY AND DRAW ITS BOX AND LABEL ---
    # This now uses the 'labels' key from your specific JSON format.
    for colony in annotation_data['labels']:
        x = colony['x']
        y = colony['y']
        width = colony['width']
        height = colony['height']

        # Define the corners of the rectangle
        x1, y1 = x, y
        x2, y2 = x + width, y + height
        
        # Draw the rectangle on the image
        draw.rectangle(
            [(x1, y1), (x2, y2)], 
            outline="red",
            width=3
        )

        # --- NEW: Draw the x and y coordinates ---
        # Create the text label
        label = f"({x}, {y})"
        # Define a position slightly above the top-left corner
        text_position = (x, y - 15)
        # Draw the text on the image
        draw.text(text_position, label, fill="yellow")
        # ----------------------------------------

    # --- 5. SHOW OR SAVE THE RESULT ---
    print("Displaying image with bounding boxes and coordinates...")
    image.show()

    output_path = 'verified_image.jpg'
    image.save(output_path)
    print(f"Annotated image saved to '{output_path}'")

except FileNotFoundError:
    print(f"Error: Could not find the image or annotation file. Check the paths.")
except KeyError:
    print("Error: The JSON file does not have the expected 'labels' key.")
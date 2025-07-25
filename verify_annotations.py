import json
from PIL import Image, ImageDraw

# --- 1. DEFINE FILE PATHS ---
# Change these to the image and annotation you want to check
image_path = 'dataset/images/123.jpg'
annotation_path = 'dataset/annotations/123.json'

try:
    # --- 2. LOAD THE IMAGE AND ANNOTATION FILE ---
    # Open the original image
    image = Image.open(image_path)
    
    # Open the JSON annotation file and load its content
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # --- 3. PREPARE TO DRAW ON THE IMAGE ---
    # Create a drawing context. All drawing happens on this 'draw' object.
    draw = ImageDraw.Draw(image)

    # --- 4. LOOP THROUGH EACH COLONY AND DRAW ITS BOX ---
    # This assumes your JSON is a dictionary with a key 'colonies' that holds a list of boxes.
    # Adjust 'annotation_data['colonies']' if your JSON structure is different.
    for colony in annotation_data['colonies']:
        # Get the coordinates and dimensions from the annotation
        x = colony['x']
        y = colony['y']
        width = colony['width']
        height = colony['height']

        # Define the corners of the rectangle
        # (x1, y1) is the top-left corner
        # (x2, y2) is the bottom-right corner
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        
        # Draw the rectangle on the image
        draw.rectangle(
            [(x1, y1), (x2, y2)], 
            outline="red",  # You can change the color (e.g., "lime", "yellow")
            width=2         # You can change the line thickness
        )

    # --- 5. SHOW OR SAVE THE RESULT ---
    # To display the image directly
    print("Displaying image with bounding boxes...")
    image.show()

    # Or, to save the annotated image to a new file
    output_path = 'verified_image.jpg'
    image.save(output_path)
    print(f"Annotated image saved to '{output_path}'")

except FileNotFoundError:
    print(f"Error: Could not find the image or annotation file. Check the paths.")
except KeyError:
    print("Error: The JSON file does not seem to have the expected format (e.g., a 'colonies' key).")
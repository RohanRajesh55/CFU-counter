import cv2
import numpy as np
import os

# Create output folder
output_dir = 'processed_output'
os.makedirs(output_dir, exist_ok=True)

# Load image
img = cv2.imread('dataset/images/337.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, '1_grayscale.jpg'), gray)

# --- New Step: Isolate Petri Dish ---
# Blur the image to reduce noise for better circle detection
blur = cv2.medianBlur(gray, 11)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(
    blur, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2, 
    minDist=gray.shape[0] // 2, # min distance between centers of circles
    param1=100, 
    param2=30, 
    minRadius=int(gray.shape[0] * 0.3), 
    maxRadius=int(gray.shape[0] * 0.5)
)

# Create a black mask with the same dimensions as the image
dish_mask = np.zeros_like(gray)

if circles is not None:
    # Convert the circle parameters (x, y, r) to integers
    circles = np.uint16(np.around(circles))
    # Get the parameters for the first (and likely only) detected circle
    x, y, r = circles[0][0]
    # Draw a white, filled circle on the mask
    cv2.circle(dish_mask, (x, y), r, 255, -1)
    cv2.imwrite(os.path.join(output_dir, 'dish_mask.jpg'), dish_mask)

    # Apply the mask to the grayscale image
    # This blacks out everything outside the white circle
    gray = cv2.bitwise_and(gray, gray, mask=dish_mask)
    cv2.imwrite(os.path.join(output_dir, '1_grayscale_isolated.jpg'), gray)
else:
    print("Warning: No petri dish circle was detected. Proceeding with the full image.")


# Step 2: Contrast stretching (more aggressive)
# Use histogram equalization instead of normalize
contrast_enhanced = cv2.equalizeHist(gray)
cv2.imwrite(os.path.join(output_dir, '2_contrast_enhanced.jpg'), contrast_enhanced)

# Step 3: Otsu's thresholding (no manual threshold value!)
_, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 4: Invert so colonies are black, background is white
inverted = cv2.bitwise_not(binary)
cv2.imwrite(os.path.join(output_dir, '3_colonies_black_background_white.jpg'), inverted)

print(f"Colonies isolated with white background. Saved in: {output_dir}")
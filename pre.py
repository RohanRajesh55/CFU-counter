import cv2
import numpy as np
import os
import pytesseract

# Paths
img_path = r"C:\Bacterial colony counter\sample1.jpg"
east_model = "frozen_east_text_detection.pb"  # Ensure this is downloaded
output_dir = "processed_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load image
img = cv2.imread(img_path)
orig = img.copy()

# Initialize EAST text detector
detector = cv2.dnn_TextDetectionModel_EAST(east_model)
detector.setConfidenceThreshold(0.5)  # Lowered to catch faint text
detector.setNMSThreshold(0.4)
detector.setInputParams(1.0, (320, 320), (123.68, 116.78, 103.94), True)

# Detect text regions
boxes, confidences = detector.detect(img)

# Create mask from EAST detections
mask = np.zeros(img.shape[:2], dtype=np.uint8)
for box in boxes:
    cv2.fillPoly(mask, [np.array(box, np.int32)], 255)

# OCR-based detection for missed text
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

for i in range(len(ocr_data['text'])):
    if int(ocr_data['conf'][i]) > 60:
        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# Save mask
cv2.imwrite(os.path.join(output_dir, "text_mask.png"), mask)

# Inpaint using TELEA
inpainted = cv2.inpaint(orig, mask, 3, cv2.INPAINT_TELEA)
cv2.imwrite(os.path.join(output_dir, "img_no_text.png"), inpainted)

# Optional: Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(inpainted, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(os.path.join(output_dir, "img_cleaned.png"), cleaned)

print("[INFO] Text removed and image cleaned successfully.")
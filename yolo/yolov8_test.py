import logging
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

# ===================================================================
# --- CONFIGURATION ---
# Edit the paths and settings below
# ===================================================================

# --- Mode 1: Full Validation Report ---
# Directory for full validation reports (metrics, images, CSV)
FULL_REPORT_OUTPUT_DIR = "runs/full_validation_report"
# Path to the image folder for the full report
SOURCE_PATH = "/home/ashwin/projects/Bacteria_colony_detection/yolo_data/images/val"
# Path to the corresponding label folder
LABELS_PATH = "/home/ashwin/projects/Bacteria_colony_detection/yolo_data/labels/val"

# --- Mode 2: Single External Image Test ---
# To run this mode, simply provide a path to an image below.
# To run the full report instead, set this to: None
TEST_IMAGE_PATH = None
# Example: TEST_IMAGE_PATH = "/home/ashwin/Pictures/new_plate_image.jpg"

# Directory for single image test results
SINGLE_IMAGE_OUTPUT_DIR = "runs/single_image_tests"


# --- General Settings ---
WEIGHTS_PATH = "runs/colony_yolo/yolov8m_optimized_run2/weights/best.pt"
DATA_YAML_PATH = "/home/ashwin/projects/Bacteria_colony_detection/yolo_data/data.yaml"
CONF_THRESHOLD = 0.40
IMG_SIZE = 1024
BATCH_SIZE = 16
FONT_SIZE = 200
COUNT_ACCURACY_BUFFERS = [#(25, 1), (100, 5), 
    (float('inf'), 10)]

# ===================================================================
# --- SCRIPT LOGIC (No need to edit below this line) ---
# ===================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class YOLOReportGenerator:
    def __init__(self, mode, **kwargs):
        self.mode = mode
        for key, value in kwargs.items(): setattr(self, key, value)
        
        # Set paths based on mode
        if self.mode == 'single_test':
            self.source_path = Path(self.test_image_path)
            self.output_dir = Path(self.single_image_output_dir)
            self.labels_path = None # No labels for a single test
        else: # full_report
            self.source_path = Path(self.source_path)
            self.labels_path = Path(self.labels_path)
            self.output_dir = Path(self.full_report_output_dir)

        self.weights_path = Path(self.weights_path)
        self.data_yaml_path = Path(self.data_yaml_path)
        self._validate_paths()
        self.model = self._load_model()
        self.class_names = self._load_class_names()
        self.custom_annotated_dir = self.output_dir / "annotated_images"
        self.custom_annotated_dir.mkdir(parents=True, exist_ok=True)

    def _validate_paths(self):
        if not self.weights_path.exists(): raise FileNotFoundError(f"Weights file not found: '{self.weights_path}'")
        if not self.data_yaml_path.exists(): raise FileNotFoundError(f"Data YAML file not found: '{self.data_yaml_path}'")
        if not self.source_path.exists(): raise FileNotFoundError(f"Source path not found: '{self.source_path}'")
        if self.mode == 'full_report' and not self.labels_path.exists(): raise FileNotFoundError(f"Labels path must be provided for full report: '{self.labels_path}'")

    def _load_model(self):
        try:
            model = YOLO(self.weights_path)
            logging.info(f"Successfully loaded model from {self.weights_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _load_class_names(self):
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config.get('names', [])

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == 'full_report':
            metrics = self.run_evaluation()
            summary_data = self.run_prediction_and_visualization()
            if summary_data:
                self._save_summary_to_csv(summary_data)
                self._log_final_summary(metrics, summary_data)
        else: # single_test
            self.run_prediction_and_visualization()
        
        logging.info(f"--- Report Generation Complete. Outputs saved in: {self.output_dir} ---")

    def run_evaluation(self):
        logging.info("--- Starting Model Evaluation (Calculating Metrics) ---")
        try:
            metrics = self.model.val(data=str(self.data_yaml_path), imgsz=self.imgsz, batch=self.batch, conf=self.conf_threshold, project=str(self.output_dir), name="metrics_and_plots", exist_ok=True, plots=True)
            return metrics
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            return None

    def run_prediction_and_visualization(self):
        logging.info("--- Starting Inference and Visual Report Generation ---")
        results_generator = self.model.predict(source=str(self.source_path), conf=self.conf_threshold, imgsz=self.imgsz, stream=True, save=False, verbose=False)
        
        summary_data = []
        total_files = 1 if self.mode == 'single_test' else len(list(self.source_path.glob('*.*')))
        
        for results in tqdm(results_generator, total=total_files, desc="Creating visual reports"):
            summary_row, annotated_image = self._process_and_draw(results)
            summary_data.append(summary_row)
            output_path = self.custom_annotated_dir / Path(results.path).name
            annotated_image.save(output_path)
        
        if self.mode == 'single_test':
            self._save_summary_to_csv(summary_data)
            
        return summary_data

    def _process_and_draw(self, results):
        image_path = Path(results.path)
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        img_width, img_height = img.size
        
        try:
            font = ImageFont.load_default(size=self.font_size)
        except AttributeError:
            font = ImageFont.load_default()

        # Get prediction data first
        predicted_detections = [self.class_names[int(box.cls[0])] for box in results.boxes]
        pred_counts = Counter(predicted_detections)
        pred_str = f"Predicted: {len(predicted_detections)} ({', '.join(f'{k}: {v}' for k, v in pred_counts.items()) or 'None'})"
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        
        summary_lines = [(pred_str, "red")]
        actual_count, count_difference = "N/A", "N/A"
        
        # Only get ground truth data if processing a directory
        if self.mode == 'full_report':
            label_path = self.labels_path / f"{image_path.stem}.txt"
            actual_detections = self._read_yolo_labels(label_path, (img_height, img_width))
            actual_counts = Counter(name for _, name in actual_detections)
            actual_str = f"Actual: {len(actual_detections)} ({', '.join(f'{k}: {v}' for k, v in actual_counts.items()) or 'None'})"
            for (x1, y1, x2, y2), _ in actual_detections:
                draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            summary_lines.insert(0, (actual_str, "green"))
            actual_count = len(actual_detections)
            count_difference = len(predicted_detections) - actual_count
        
        # Draw summary text in the top-right
        y_offset = 10
        for text, color in summary_lines:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x = img_width - text_width - 15
            text_y = y_offset
            draw.rectangle([(text_x - 5, text_y - 5), (img_width, text_y + text_height + 5)], fill=(0, 0, 0, 150))
            draw.text((text_x, text_y), text, fill=color, font=font)
            y_offset += text_height + 10
        
        summary_row = {
            "Image Name": image_path.name, "Actual Count": actual_count, 
            "Predicted Count": len(predicted_detections), "Count Difference": count_difference,
            "Predicted Classes": ', '.join(f'{k}: {v}' for k, v in pred_counts.items()) or 'None'
        }
        return summary_row, img

    def _read_yolo_labels(self, label_file_path, image_shape):
        detections = []
        if not label_file_path.exists(): return detections
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls_id, (img_height, img_width) = int(parts[0]), image_shape
                center_x, center_y, w, h = parts[1:]
                x1, y1 = int((center_x - w / 2) * img_width), int((center_y - h / 2) * img_height)
                x2, y2 = int((center_x + w / 2) * img_width), int((center_y + h / 2) * img_height)
                detections.append(((x1, y1, x2, y2), self.class_names[cls_id]))
        return detections
        
    def _save_summary_to_csv(self, summary_data):
        if not summary_data: return
        df = pd.DataFrame(summary_data)
        column_order = ["Image Name", "Actual Count", "Predicted Count", "Count Difference", "Predicted Classes"]
        if self.mode == 'single_test':
            column_order = ["Image Name", "Predicted Count", "Predicted Classes"]
        df = df[column_order]
        csv_path = self.output_dir / "detection_summary.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Successfully created detection summary at: {csv_path}")

    def _log_final_summary(self, metrics, summary_data):
        if not summary_data: return
        logging.info("--- FINAL PERFORMANCE SUMMARY ---")
        correct_strict, correct_buffered = 0, 0
        for row in summary_data:
            if row["Count Difference"] == 0: correct_strict += 1
            actual_count, abs_diff = row["Actual Count"], abs(row["Count Difference"])
            allowed_buffer = 0
            for upper_bound, buffer_val in self.count_accuracy_buffers:
                if actual_count < upper_bound:
                    allowed_buffer = buffer_val
                    break
            if abs_diff <= allowed_buffer: correct_buffered += 1
        total_images = len(summary_data)
        strict_accuracy = (correct_strict / total_images) * 100 if total_images > 0 else 0
        buffered_accuracy = (correct_buffered / total_images) * 100 if total_images > 0 else 0
        logging.info(f"Exact Count Accuracy       : {strict_accuracy:.2f} % ({correct_strict}/{total_images})")
        logging.info(f"Buffered Count Accuracy     : {buffered_accuracy:.2f} % ({correct_buffered}/{total_images})")
        actual_counts = [row["Actual Count"] for row in summary_data]
        predicted_counts = [row["Predicted Count"] for row in summary_data]
        try:
            pseudo_r_squared = np.corrcoef(actual_counts, predicted_counts)[0, 1]**2
            logging.info(f" R-squared (Counts)   : {pseudo_r_squared:.4f}")
        except Exception:
            logging.warning("Pseudo R-squared could not be calculated.")
        if metrics:
            classification_accuracy = metrics.box.p[0] * 100
            logging.info(f"Classification Accuracy (P)   : {classification_accuracy:.2f} %")
            logging.info(f"Overall Recall                : {metrics.box.r[0]:.3f}")
            logging.info(f"Overall mAP50                 : {metrics.box.map50:.3f}")
            logging.info(f"Overall mAP50-95              : {metrics.box.map:.3f}")

if __name__ == "__main__":
    # Determine mode based on TEST_IMAGE_PATH
    mode = 'single_test' if TEST_IMAGE_PATH else 'full_report'
    
    config = {
        'weights_path': WEIGHTS_PATH, 'data_yaml_path': DATA_YAML_PATH,
        'source_path': SOURCE_PATH, 'labels_path': LABELS_PATH,
        'test_image_path': TEST_IMAGE_PATH,
        'full_report_output_dir': FULL_REPORT_OUTPUT_DIR,
        'single_image_output_dir': SINGLE_IMAGE_OUTPUT_DIR,
        'conf_threshold': CONF_THRESHOLD, 'imgsz': IMG_SIZE,
        'batch': BATCH_SIZE, 'font_size': FONT_SIZE,
        'count_accuracy_buffers': COUNT_ACCURACY_BUFFERS
    }
    
    try:
        report_generator = YOLOReportGenerator(mode=mode, **config)
        report_generator.run()
    except Exception as e:
        logging.error(f"Failed to initialize or run the report generator: {e}", exc_info=True)
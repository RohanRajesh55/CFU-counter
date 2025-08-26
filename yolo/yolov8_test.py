import logging
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# ===================================================================
# --- CONFIGURATION ---
# Edit the paths and settings below
# ===================================================================

# Path to your trained model weights
WEIGHTS_PATH = "runs/colony_yolo/yolov8m_optimized_run2/weights/best.pt"
# Path to your dataset's .yaml configuration file (needed for metrics)
DATA_YAML_PATH = "/home/ashwin/projects/Bacteria_colony_detection/yolo_data/data.yaml"
# Path to the image folder OR a single image file you want to evaluate
SOURCE_PATH = "/home/ashwin/projects/Bacteria_colony_detection/yolo_data/images/val"
# Path to the corresponding label folder for ground truth counts
LABELS_PATH = "/home/ashwin/projects/Bacteria_colony_detection/yolo_data/labels/val"
# Main directory to save all outputs
OUTPUT_DIR = "runs/final_report"

# --- Model & Evaluation Settings ---
CONF_THRESHOLD = 0.40
IMG_SIZE = 1024
BATCH_SIZE = 16

# --- NEW: Buffer settings for Count Accuracy ---
# Format: (upper_bound, allowed_buffer)
# The script checks these rules in order.
COUNT_ACCURACY_BUFFERS = [
    (25, 1),    # For actual counts < 25, allow a buffer of +/- 1
    (100, 5),   # For actual counts < 100, allow a buffer of +/- 5
    (float('inf'), 10) # For all other counts (100+), allow a buffer of +/- 10
]

# ===================================================================
# --- SCRIPT LOGIC (No need to edit below this line) ---
# ===================================================================

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class YOLOReportGenerator:
    def __init__(self, **kwargs):
        # Assign all kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.weights_path = Path(self.weights_path)
        self.data_yaml_path = Path(self.data_yaml_path)
        self.source_path = Path(self.source_path)
        self.labels_path = Path(self.labels_path)
        self.output_dir = Path(self.output_dir)
        self._validate_paths()
        self.model = self._load_model()
        
    def _validate_paths(self):
        if not self.weights_path.exists(): raise FileNotFoundError(f"Weights file not found: '{self.weights_path}'")
        if not self.data_yaml_path.exists(): raise FileNotFoundError(f"Data YAML file not found: '{self.data_yaml_path}'")
        if not self.source_path.exists(): raise FileNotFoundError(f"Source path not found: '{self.source_path}'")
        if self.source_path.is_dir() and not self.labels_path.exists(): raise FileNotFoundError(f"Labels path must be provided: '{self.labels_path}'")

    def _load_model(self):
        try:
            model = YOLO(self.weights_path)
            logging.info(f"Successfully loaded model from {self.weights_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def run_evaluation(self):
        logging.info("--- Starting Model Evaluation (Calculating Metrics) ---")
        try:
            metrics = self.model.val(data=str(self.data_yaml_path), imgsz=self.imgsz, batch=self.batch, conf=self.conf_threshold, project=str(self.output_dir), name="metrics_and_plots", exist_ok=True, plots=True)
            logging.info(f"Metrics and plots saved to '{self.output_dir}/metrics_and_plots'")
            return metrics
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            return None

    def run_prediction_and_csv(self):
        logging.info("--- Starting Inference and Report Generation ---")
        try:
            results_generator = self.model.predict(source=str(self.source_path), conf=self.conf_threshold, imgsz=self.imgsz, stream=True, save=True, project=str(self.output_dir), name="annotated_images", exist_ok=True)
            summary_data = []
            is_dir = self.source_path.is_dir()
            total_files = len(list(self.source_path.glob('*.*'))) if is_dir else 1
            for results in tqdm(results_generator, total=total_files, desc="Processing images for CSV report"):
                summary_data.append(self._process_single_result(results))
            self._save_summary_to_csv(summary_data)
            return summary_data
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            return []

    def _process_single_result(self, results):
        image_name, boxes = Path(results.path).name, results.boxes
        predicted_count = len(boxes)
        actual_count, count_difference = "N/A", "N/A"
        if self.source_path.is_dir():
            label_path = self.labels_path / f"{Path(image_name).stem}.txt"
            count = 0
            if label_path.exists():
                with open(label_path) as f: count = len(f.readlines())
            actual_count = count
            count_difference = predicted_count - actual_count
        details_str = "None"
        if predicted_count > 0:
            class_confidences = defaultdict(list)
            for cls_id, conf in zip(boxes.cls.int().tolist(), boxes.conf.tolist()):
                class_name = self.model.names.get(cls_id, f"UnknownID_{cls_id}")
                class_confidences[class_name].append(conf)
            detection_summary = {name: (len(conf_list), sum(conf_list) / len(conf_list)) for name, conf_list in class_confidences.items()}
            details_str = ", ".join([f"{name}: {count} | {avg_conf:.2f}" for name, (count, avg_conf) in detection_summary.items()])
        return {"Image Name": image_name, "Actual Count": actual_count, "Predicted Count": predicted_count,
                "Count Difference": count_difference, "Detections": details_str}
        
    def _save_summary_to_csv(self, summary_data):
        if not summary_data: return
        df = pd.DataFrame(summary_data)
        column_order = ["Image Name", "Actual Count", "Predicted Count", "Count Difference", "Detections"]
        if self.source_path.is_file():
            column_order = [col for col in column_order if col not in ["Actual Count", "Count Difference"]]
        df = df[column_order]
        csv_path = self.output_dir / "detection_summary.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Successfully created detection summary at: {csv_path}")
        
    def _log_final_summary(self, metrics, summary_data):
        if not self.source_path.is_dir() or not summary_data: return
        
        logging.info("--- FINAL PERFORMANCE SUMMARY ---")
        
        # --- NEW: Buffered Accuracy Calculation ---
        correct_strict, correct_buffered = 0, 0
        for row in summary_data:
            if row["Count Difference"] == 0:
                correct_strict += 1
            
            actual_count = row["Actual Count"]
            abs_diff = abs(row["Count Difference"])
            
            # Determine allowed buffer based on rules
            allowed_buffer = 0
            for upper_bound, buffer_val in self.count_accuracy_buffers:
                if actual_count < upper_bound:
                    allowed_buffer = buffer_val
                    break
            
            if abs_diff <= allowed_buffer:
                correct_buffered += 1
        
        total_images = len(summary_data)
        strict_accuracy = (correct_strict / total_images) * 100 if total_images > 0 else 0
        buffered_accuracy = (correct_buffered / total_images) * 100 if total_images > 0 else 0
        
        logging.info(f"Strict Count Accuracy       : {strict_accuracy:.2f} % ({correct_strict}/{total_images} images have exact counts)")
        logging.info(f"Buffered Count Accuracy     : {buffered_accuracy:.2f} % ({correct_buffered}/{total_images} images are within tolerance)")
        
        # --- Pseudo R-squared for counts ---
        actual_counts = [row["Actual Count"] for row in summary_data]
        predicted_counts = [row["Predicted Count"] for row in summary_data]
        try:
            correlation_matrix = np.corrcoef(actual_counts, predicted_counts)
            pseudo_r_squared = correlation_matrix[0, 1]**2
            logging.info(f"Pseudo R-squared (Counts)   : {pseudo_r_squared:.4f}")
        except (np.linalg.LinAlgError, ValueError):
            logging.warning("Pseudo R-squared could not be calculated.")

        # --- Standard object detection metrics ---
        if metrics:
            logging.info(f"Overall mAP50-95              : {metrics.box.map:.3f}")
            logging.info(f"Overall mAP50                 : {metrics.box.map50:.3f}")
            logging.info(f"Overall Precision             : {metrics.box.p[0]:.3f}")
            logging.info(f"Overall Recall                : {metrics.box.r[0]:.3f}")

    def generate_full_report(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics = None
        if self.source_path.is_dir():
            metrics = self.run_evaluation()
        summary_data = self.run_prediction_and_csv()
        if self.source_path.is_dir():
            self._log_final_summary(metrics, summary_data)
        logging.info("--- Full Report Generation Complete ---")

if __name__ == "__main__":
    try:
        config = {
            'weights_path': WEIGHTS_PATH, 'data_yaml_path': DATA_YAML_PATH,
            'source_path': SOURCE_PATH, 'labels_path': LABELS_PATH,
            'output_dir': OUTPUT_DIR, 'conf_threshold': CONF_THRESHOLD,
            'imgsz': IMG_SIZE, 'batch': BATCH_SIZE,
            'count_accuracy_buffers': COUNT_ACCURACY_BUFFERS
        }
        report_generator = YOLOReportGenerator(**config)
        report_generator.generate_full_report()
    except Exception as e:
        logging.error(f"Failed to initialize or run the report generator: {e}", exc_info=True)
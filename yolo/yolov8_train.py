import argparse
import yaml
import logging
import sys
from pathlib import Path
from ultralytics import YOLO

# Set up professional logging to provide clear feedback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Define keys that are part of the dataset YAML but are NOT training arguments ---
DATASET_KEYS = ['path', 'train', 'val', 'test', 'nc', 'names']

def train_model(config_path: str):
    """
    Loads a configuration from a YAML file and trains a YOLOv8 model.
    This version correctly separates dataset keys from training hyperparameters.
    """
    # 1. Load the training configuration from the YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded training configuration from {config_path}")
    except Exception as e:
        logging.error(f"Error loading or parsing YAML file: {e}")
        return

    # 2. Initialize the YOLO model
    try:
        model = YOLO(config.get('model', 'yolov8n.pt'))
        logging.info(f"Model '{config.get('model', 'yolov8n.pt')}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        return

    # 3. Create a dictionary of training arguments, filtering out the dataset keys
    train_args = {k: v for k, v in config.items() if k not in DATASET_KEYS}

    # 4. Start the training process
    logging.info("Starting model training...")
    try:
        # Pass the config file path to the 'data' argument, and the rest as keyword arguments
        model.train(data=config_path, **train_args)
        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        logging.error("Training was halted. Please check your configuration and data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model using a YAML configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file (e.g., data.yaml)."
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.is_file():
        logging.error(f"The specified config file does not exist: {args.config}")
    else:
        train_model(args.config)
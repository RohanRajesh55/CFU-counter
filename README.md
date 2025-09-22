---
# CFU‑Counter: Automated Colony Counting on Agar Plates

## Introduction

Counting bacterial colonies on agar plates is a fundamental task in microbiology, but manual counting is slow, tedious, and prone to error. This project develops a **deep learning–based system** to automatically count and classify Colony Forming Units (CFU) from digital images, providing both speed and reproducibility.
---

## Dataset

### Source

We use the **AGAR dataset** (Annotated Germs for Automated Recognition) provided by NeuroSYS — a comprehensive public dataset for microbiology computer vision tasks.

### Components

Each sample includes:

- **Image:** High‑resolution photo of a petri dish, spanning diverse colony types, densities (empty to confluent), and lighting conditions.
- **Annotation (.json):**
  - `"colonies_number"`: True colony count
  - `"classes"`: Plate‑level colony type(s)
  - `"labels"`: Bounding box coordinates + class for each colony

### Preprocessing

- Validated JSON integrity, normalized class names
- Cropped/letterboxed images to dish region
- Stratified train/val/test splits by class and density

---

## Methodology

We explored two strategies:

### 1. Custom Multi‑Task CNN

- **Architecture:** 5 convolutional blocks with BatchNorm → shared embedding → two heads:
  - **Regression head:** Predicts total colony count
  - **Classification head:** Predicts plate‑level class
- **Loss:** MSE (count) + Cross‑Entropy (class)
- **Training:** Adam optimizer, ReduceLROnPlateau, early stopping
- **Augmentation:** Flips, rotations, mild color jitter

### 2. YOLOv8 Object Detection

- **Formulation:** Detect each colony with bounding box, class, and confidence; CFU = number of detections
- **Model:** Fine‑tuned YOLOv8m on AGAR
- **Advantages:**
  - Per‑colony localization and class
  - Handles mixed cultures
  - Pretrained backbone → higher accuracy and robustness

---

## Results

| Metric                            | Score      |
| --------------------------------- | ---------- |
| **mAP@.5:.95**                    | **0.622**  |
| **Precision**                     | **0.981**  |
| **Recall**                        | **0.966**  |
| **Count Accuracy**                | **69.49%** |
| **Buffered Count Accuracy (±5%)** | **88.73%** |

### Key Findings

- **High precision (98.1%)** → very few false positives
- **Buffered accuracy (88.7%)** → counts within ±5% of ground truth for most plates
- **YOLOv8 outperforms CNN** by providing per‑colony explainability and robustness to mixed cultures

---

## Error Analysis

- **High density:** Misses or double‑boxes in confluent regions
- **Low contrast colonies:** Missed under glare/shadows
- **Plate edges:** Rim reflections and condensation → false positives
- **Rare classes:** Lower AP due to imbalance

**Mitigations:** dish masking, higher resolution input, tuned NMS, focal loss for class imbalance.

---

## CNN vs YOLOv8

| Aspect              | CNN                 | YOLOv8                   |
| ------------------- | ------------------- | ------------------------ |
| Output              | Plate count + class | Per‑colony boxes + class |
| Explainability      | Low                 | High                     |
| Mixed cultures      | Not supported       | Supported                |
| Density sensitivity | High                | Moderate                 |
| Deployment          | Very light          | Real‑time capable        |

---

## Reproducibility

- **Config versioning:** YAML configs for data, augmentation, model, evaluation
- **Evaluation protocol:**
  - Detection: COCO‑style mAP@.5:.95
  - Counting: raw accuracy + buffered accuracy at ±2%, ±5%, ±10%
- **Reporting:** Per‑class AP, PR curves, density‑binned results, Bland–Altman plots

---

## Limitations & Next Steps

- **Limitations:** Confluent growth, illumination variability, rare phenotypes
- **Future work:**
  - Hybrid detection + segmentation for merged colonies
  - Flat‑field correction for lighting artifacts
  - Curriculum training by density
  - Uncertainty‑aware counts with confidence intervals

---

## Citation

If you use this project, please cite the AGAR dataset and this repository.

---

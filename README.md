### ## Introduction

This project aims to automate the process of counting bacteria colonies on agar plates, a fundamental task in microbiology that is traditionally performed manually. Manual counting is slow, tedious, and prone to human error. Our goal is to develop a robust system using deep learning to accurately count and classify the number of Colony Forming Units (CFU) from a digital image.

---

### ## Dataset

#### Source

The project utilizes the **AGAR dataset** from _The Annotated Germs for Automated Recognition_, provided by NeuroSYS. It's a comprehensive public dataset designed for computer vision tasks in microbiology.

#### Components

Each sample in the dataset consists of two parts:

1.  **Image:** A high-resolution digital photograph of a petri dish. The images feature a wide variety of bacteria types, colony densities (from empty to uncountable), and lighting conditions.
2.  **Annotation File:** A corresponding `.json` file that contains the ground truth information, including:
    - `"colonies_number"`: The true total count of colonies.
    - `"classes"`: The type of bacteria present on the plate.
    - `"labels"`: A list of **bounding box coordinates** for each individual colony, which is critical for the object detection approach.

---

### ## Methodology & Models Explored

To tackle this problem, we investigated two distinct deep learning strategies: a custom multi-task CNN and a state-of-the-art object detection model.

#### Approach 1: Custom CNN for Multi-Task Learning

Our initial approach involved a multi-task learning framework using a custom Convolutional Neural Network built from scratch in PyTorch.

- **Model:** A deep CNN was designed with 5 convolutional blocks. Each block uses Batch Normalization for stable training. The network has two separate output heads:
  1.  A **Regression Head** to predict the total colony count.
  2.  A **Classification Head** to predict the overall colony type for the plate.
- **Training:** The model was trained to minimize a combined loss function (Mean Squared Error for the count and Cross-Entropy for the type), optimized with an Adam optimizer, a `ReduceLROnPlateau` scheduler, and Early Stopping.
- **Data Augmentation:** To improve generalization, the training data was augmented with random flips, rotations, and color jitter.

#### Approach 2: YOLOv8 for Object Detection (Advanced Method)

To achieve a more granular and powerful analysis, we reframed the problem from simple regression to **object detection**. This approach aims to locate and classify _every single colony_ on the plate.

- **Model:** We selected **YOLOv8**, a state-of-the-art, single-stage object detector known for its exceptional speed and accuracy. The pre-trained `yolov8m` (medium) variant was fine-tuned on the AGAR dataset.
- **Task Formulation:** Instead of predicting a single count, YOLOv8 processes an image and outputs a list of bounding boxes. For each box, it provides:
  1.  The **class** of the individual colony (e.g., `E.coli`, `Defect`).
  2.  The **coordinates** of the colony on the plate.
  3.  A **confidence score** for the detection.
      The final colony count is then simply the total number of boxes detected.
- **Advantages over Custom CNN:**
  - **Per-Colony Information:** Provides the location and class of each colony, not just a total.
  - **Handles Mixed Cultures:** Can detect and count multiple different classes within the same image, which a single-label classification head cannot.
  - **Superior Accuracy:** Leverages a powerful pre-trained architecture, which led to significantly higher performance in our experiments.

---

### ## Results of YOLOv8 Approach

The fine-tuned YOLOv8 model demonstrated outstanding performance, proving to be a highly effective solution.

| Metric                      | Score      |
| :-------------------------- | :--------- |
| **Overall mAP@.5:.95**      | **0.622**  |
| **Overall Precision**       | **0.981**  |
| **Overall Recall**          | **0.966**  |
| **Buffered Count Accuracy** | **98.27%** |

**Key Findings:**

- The model achieved an exceptional **Precision of 98.1%**, indicating that its detections are highly reliable with very few false positives.
- The **Buffered Count Accuracy of 98.27%** shows that for nearly all images, the model's total count is within a small, scientifically acceptable margin of error, making it a robust tool for quantitative analysis.

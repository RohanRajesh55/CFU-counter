# CFU-counter

## 📖 Introduction

This project aims to automate the process of counting bacteria colonies on agar plates, a fundamental task in microbiology that is traditionally performed manually. Manual counting is slow, tedious, and prone to human error. Our goal is to develop a robust system using deep learning, specifically Convolutional Neural Networks (CNNs), to accurately count the number of Colony Forming Units (CFU) from a digital image.

---

## 📊 Dataset

### Source

The project utilizes the **AGAR dataset** from _The Annotated Germs for Automated Recognition_, provided by NeuroSYS. It's a comprehensive public dataset designed for computer vision tasks in microbiology.

### Components

Each sample in the dataset consists of two parts:

1.  **Image:** A high-resolution digital photograph of a petri dish. The images feature a wide variety of bacteria types, colony densities (from empty to uncountable), and lighting conditions.
2.  **Annotation File:** A corresponding `.json` file that contains the ground truth information, including:
    - `"colonies_number"`: The true total count of colonies (our regression target).
    - `"classes"`: The type of bacteria (our classification target).
    - `"labels"`: A list of bounding box coordinates for each individual colony.

---

## 🔬 Methodology

We explored a multi-task learning approach using a custom Convolutional Neural Network built from scratch using PyTorch.

- **Model:** A deep CNN was designed with 5 convolutional blocks. Each block uses Batch Normalization for stable training. The network has two separate output heads:
  1.  A **Regression Head** to predict the colony count.
  2.  A **Classification Head** to predict the colony type.
- **Training:** The model is trained to minimize a combined loss function (Mean Squared Error for the count and Cross-Entropy for the type). The training process is optimized with an Adam optimizer, a `ReduceLROnPlateau` learning rate scheduler, and Early Stopping.
- **Data Augmentation:** To improve generalization, the training data is augmented with random flips, rotations, and color jitter.

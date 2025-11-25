# Fashion-MNIST-PyTorch-CNN
A foundational CNN implementation in PyTorch for classifying Fashion MNIST images.
# 👕 Fashion MNIST Image Classifier: A PyTorch CNN Implementation

### Overview

This project implements a foundational **Convolutional Neural Network (CNN)** using the **PyTorch** framework to classify images of clothing from the Fashion MNIST dataset. The objective is to construct a simple yet effective deep learning model that automatically learns distinguishing features from 28x28 grayscale images of apparel (e.g., T-shirts, trousers, and coats). This serves as a complete, working example of the modern deep learning pipeline.

### Features

* **Custom CNN Architecture:** A sequential model defined in PyTorch with two convolutional layers for feature extraction and a fully-connected classifier head.
* **Fashion MNIST Dataset:** Utilizes `torchvision.datasets` for easy access to 60,000 training and 10,000 testing images.
* **Standardized Preprocessing:** Images are transformed into PyTorch Tensors and normalized for stable training.
* **Device Agnostic:** Training automatically detects and utilizes a **CUDA-enabled GPU** if available, ensuring performance optimization.
* **Modular Training Loop:** Includes separate, clear functions for training, testing, and the main execution loop, tracking loss and accuracy.

### Technologies/Tools Used

* **Deep Learning Framework:** PyTorch (v1.10+)
* **Computer Vision/Data:** `torchvision`
* **Utility:** `numpy`, `matplotlib`

### ⚙️ Steps to Install & Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Patil-s16/Fashion-MNIST-PyTorch-CNN]
    cd Fashion-MNIST-PyTorch-CNN
    ```

2.  **Install Dependencies:**
    The project relies on PyTorch and standard scientific computing libraries. Run the following command (or the first cell in the notebook) to install everything:
    ```bash
    pip install -q torch torchvision matplotlib numpy
    ```
    *(Note: If you run into issues, refer to the official PyTorch website for the correct installation command for your specific OS/CUDA version.)*

3.  **Run the Notebook:**
    Open the `cnn_project.ipynb` file in a Jupyter environment (JupyterLab, VS Code, or Google Colab) and run all cells sequentially.
    * The data will be downloaded automatically.
    * The model will be defined.
    * The training will commence, logging progress per epoch.

### 🧪 Instructions for Testing

The model is evaluated using the dedicated, unseen test dataset (`test_dataset` with 10,000 images).

1.  **Integrated Evaluation:** The main training loop automatically calculates and reports performance on the test set after every training epoch.
2.  **Performance Metrics:** Review the output logs for the `Test Acc` metric.
3.  **Success Criterion:** A successful implementation typically achieves a **Test Accuracy** of **88% or higher**.

---

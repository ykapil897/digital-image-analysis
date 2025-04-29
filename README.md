# Digital Image Analysis Projects

This repository contains several digital image analysis projects demonstrating various image processing techniques and transformations.

---

## 📁 Contents

- [Assignment 1: Basic Image Transformations](./assign_1.py)  
- [Assignment 2: Spatial Filtering Techniques](./assign_2/assign_2.py)  
- [Assignment 3: Frequency Domain Filtering](./assign_3/assign_3.py)  
- [Project: Texture Classification](./project/proj.ipynb)


---

## 🖼 Assignment 1: Basic Image Transformations

This assignment implements fundamental pixel-level transformations to understand their effects on grayscale images.

### Techniques Implemented
- **Log Transformation**: Enhances contrast in darker regions while compressing brighter ones.
- **Gamma Transformation**: Adjusts brightness and contrast via power-law.
- **Piecewise Linear Transformation**: Enhances contrast in specified intensity ranges.
- **Bit Plane Slicing**: Visualizes individual bit-plane contributions.
- **Histogram Equalization**: Redistributes intensities for better contrast.
- **Combined Transformations**: Applies multiple transformations sequentially.

### 📌 Usage
Ensure a grayscale image named `Grayscale_8bits_palette_sample_image.png` is present in the same directory.

---

## 🧮 Assignment 2: Spatial Filtering Techniques

This assignment implements spatial domain filters for image enhancement and restoration.

### Techniques Implemented
- **Mean Filtering** (correlation & convolution)
- **Gaussian Filtering** with varying kernel sizes
- **Median Filtering** to remove salt-and-pepper noise
- **Sobel Filters** for gradient and edge detection
- **Laplacian Filtering** for edge sharpening
- **Laplacian of Gaussian (LoG)** for smoothed edge detection
- **Unsharp Masking & Highboost Filtering** for sharpening
- **Combined Filters** applying multiple sequentially

### 📂 Sample Results
Results saved in the `output/` directory include:
- Original and noisy images
- Filtered outputs (mean, Gaussian, median)
- Edge maps from Sobel, Laplacian, and LoG
- Sharpened images via unsharp masking & highboost

---

## 🌐 Assignment 3: Frequency Domain Filtering

Focuses on filtering in the frequency domain to remove periodic noise.

### Techniques Implemented
- **Notch Filtering** to suppress periodic noise
- **Butterworth Notch Filter** for smoother filtering transitions

---

## 🧵 Project: Texture Classification

Builds a texture classification system using the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

### Feature Extraction
- **GLCM (Gray-Level Co-occurrence Matrix)**
- **Gabor Filters**

### Classification Algorithms
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Convolutional Neural Network (CNN)**

### ⚙️ Implementation
Implemented in a Jupyter Notebook: `proj.ipynb`, which includes:
- Data loading and preprocessing
- Feature extraction
- Training and evaluation
- ML vs DL performance comparison

### 📌 Usage
- Download and extract DTD into folder `dtd-r1.0.1`
- Run the notebook in the same directory

---

## 📦 Requirements

- Python 3.6+
- NumPy
- OpenCV
- Matplotlib
- scikit-image
- SciPy
- TensorFlow/Keras
- Jupyter Notebook

Install all dependencies:
```bash
pip install numpy opencv-python matplotlib scikit-image scipy tensorflow jupyter
```
# 🧠 Brain Tumor Detection using Deep Learning

An end-to-end **Brain Tumor Classification System** built using **Deep Learning, Transfer Learning, PyTorch, ResNet50, and Flask**.

The application analyzes brain MRI images and classifies them into three tumor categories:

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Vicky8021/Brain-Tumor-Detection)

- 🔬 **Glioma**
- 🔬 **Meningioma**
- 🔬 **Pituitary Tumor**

The trained model achieved **94.05% accuracy on the held-out test dataset**.

---

## 🚀 Project Overview

Brain tumor classification from MRI scans is an important Computer Vision and Deep Learning problem.

This project implements a complete machine learning pipeline from raw `.mat` dataset files to a Flask web application and cloud deployment.

### End-to-End Pipeline

```text
Brain MRI Dataset
        ↓
MATLAB (.mat) Data Extraction
        ↓
Image Preprocessing
        ↓
Patient-Level Dataset Splitting
        ↓
ResNet50 Transfer Learning
        ↓
Model Fine-Tuning
        ↓
Model Evaluation
        ↓
Trained Model
        ↓
Hugging Face Model Hub
        ↓
Flask Web Application
        ↓
Render Deployment
        ↓
MRI Image Upload
        ↓
Tumor Prediction
        ↓
Confidence Score
```

The project is designed primarily for **educational and research purposes**.

---

# ✨ Features

- 🧠 Brain MRI image classification
- 🔬 3-class tumor classification
- ⚡ ResNet50 transfer learning
- 🎯 **94.05% test accuracy**
- 🏆 **95.84% best validation accuracy**
- 📊 Precision, Recall and F1-score evaluation
- 🔀 Confusion matrix evaluation
- 👨‍💻 Flask-based web application
- 📤 MRI image upload
- 🤖 AI-powered tumor prediction
- 📈 Prediction confidence display
- 📱 Responsive web interface
- 🔒 Patient-level dataset splitting
- 🛡️ Large training data excluded from GitHub
- 🤗 Trained model hosted on Hugging Face
- 🚀 Render deployment ready
- 💻 GPU training using Google Colab Tesla T4

---

# 🧠 Tumor Classes

The model performs classification between three tumor categories:

| Class | Description |
|---|---|
| **Glioma** | Brain tumor originating from glial cells |
| **Meningioma** | Tumor arising from the meninges |
| **Pituitary** | Tumor involving the pituitary gland |

---

# 🗂️ Dataset

The project uses a brain tumor MRI dataset originally stored in MATLAB `.mat` format.

## Dataset Statistics

```text
Total MRI Images:       3,064
Unique Patients:          233

Training Images:        2,153
Validation Images:        457
Test Images:              454

Training Patients:        163
Validation Patients:       35
Test Patients:             35
```

## Class Distribution

### Training Set

```text
Glioma        980
Pituitary     659
Meningioma    514
```

### Validation Set

```text
Glioma        236
Meningioma    111
Pituitary     110
```

### Test Set

```text
Glioma        210
Pituitary     161
Meningioma     83
```

---

# 🔐 Patient-Level Dataset Splitting

A major part of this project is preventing **patient-level data leakage**.

Instead of randomly splitting individual MRI images, the dataset was split using **patient IDs**.

This ensures that images belonging to the same patient are not distributed across training, validation, and test sets.

## Patient Split

```text
Train Patients:       163
Validation Patients:   35
Test Patients:         35
```

## Patient Overlap

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

This provides a more reliable evaluation of how the model performs on unseen patients.

---

# ⚙️ Data Preprocessing

The original dataset contains MRI images stored inside MATLAB `.mat` files.

The preprocessing pipeline performs the following steps:

1. Reads `.mat` files using `h5py`
2. Extracts MRI image data
3. Extracts tumor labels
4. Maps numeric labels to tumor classes
5. Normalizes pixel values
6. Converts images into JPEG format
7. Organizes images by tumor class
8. Creates dataset metadata
9. Performs patient-level train/validation/test splitting

The processed dataset follows this structure:

```text
processed_dataset/
│
├── Glioma/
├── Meningioma/
└── Pituitary/
```

The complete processed dataset is **not included in this GitHub repository** because of its size.

---

# 🧠 Deep Learning Model

## ResNet50 Transfer Learning

The project uses **ResNet50**, a deep Convolutional Neural Network architecture.

Instead of training the model completely from scratch, **Transfer Learning** is used.

A pre-trained ResNet50 model is adapted and fine-tuned for brain MRI tumor classification.

```text
Pre-trained ResNet50
        ↓
Transfer Learning
        ↓
Brain MRI Dataset
        ↓
Fine-Tuning
        ↓
3-Class Classification
```

## Model Architecture

The final classification head uses:

```text
ResNet50 Backbone
       ↓
2048 Features
       ↓
Linear Layer
2048 → 512
       ↓
ReLU
       ↓
Dropout
       ↓
Linear Layer
512 → 3
       ↓
Tumor Class
```

### Why Transfer Learning?

Transfer learning allows the model to start with previously learned visual features and adapt them to the brain MRI classification task.

This can provide:

- Faster training
- Better feature extraction
- Improved performance
- Reduced requirement for training from scratch

---

# ⚙️ Training Configuration

```text
Model:               ResNet50
Framework:           PyTorch
GPU:                 Tesla T4
Batch Size:          32
Epochs:              25
Warm-up Epochs:       2
Number of Classes:   3
```

Training was performed using **Google Colab with a Tesla T4 GPU**.

---

# 📊 Model Performance

## 🎯 Final Test Accuracy

```text
94.05%
```

## 🏆 Best Validation Accuracy

```text
95.84%
```

The model was evaluated on a completely held-out test set containing **454 MRI images from 35 patients**.

---

# 📋 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Meningioma | 90.14% | 77.11% | 83.12% | 83 |
| Glioma | 91.40% | 96.19% | 93.74% | 210 |
| Pituitary | 99.38% | 100.00% | 99.69% | 161 |
| **Accuracy** | | | **94.05%** | **454** |
| Macro Avg | 93.64% | 91.10% | 92.18% | 454 |
| Weighted Avg | 94.00% | 94.05% | 93.91% | 454 |

---

# 🔀 Confusion Matrix

```text
                 Predicted
              Men   Gli   Pit

Actual Men     64    19     0
Actual Gli      7   202     1
Actual Pit      0     0   161
```

The model achieved particularly strong performance on the **Pituitary** class.

The largest classification confusion occurred between **Meningioma and Glioma**.

---

# 📈 Training Progress

Selected validation accuracy results during training:

```text
Epoch 01  → 68.71%
Epoch 02  → 85.34%
Epoch 04  → 93.22%
Epoch 10  → 94.97%
Epoch 12  → 95.40%
Epoch 20  → 95.84%
```

### Best Validation Accuracy

```text
95.84%
```

### Final Test Accuracy

```text
94.05%
```

---

# 🌐 Flask Web Application

The trained model is integrated into a **Flask web application**.

The application allows a user to upload a brain MRI image and receive a predicted tumor class and confidence score.

## Application Workflow

```text
User
 ↓
Upload MRI Image
 ↓
Flask Backend
 ↓
Image Preprocessing
 ↓
ResNet50 Model
 ↓
Softmax Prediction
 ↓
Tumor Class + Confidence
 ↓
Prediction Result
```

---

# 🖥️ Application Pages

The Flask application contains:

### 🏠 Home Page

Provides:

- Project introduction
- Model information
- Accuracy information
- Detection button
- Developer information
- GitHub profile

### 📤 Upload Page

Allows the user to upload an MRI image.

### 🔍 Prediction Page

Displays:

- Predicted tumor class
- Model confidence
- Uploaded MRI image

### ⚠️ Error Page

Handles invalid uploads and application errors gracefully.

---

# 🤗 Hugging Face Model Hosting

The trained ResNet50 model is hosted separately from the GitHub source repository.

## Model Repository

**Hugging Face Model:**

https://huggingface.co/Vicky8021/brain-tumor-resnet50

The repository contains:

```text
bt_resnet50_model.pt
```

The trained model is approximately **99 MB**.

### Why Hugging Face?

The trained `.pt` file is intentionally excluded from GitHub to keep the source repository lightweight.

The Flask application uses the `huggingface_hub` library to automatically download the trained model when the application starts.

### Model Loading Flow

```text
Flask Application Starts
        ↓
Hugging Face Hub
        ↓
Download bt_resnet50_model.pt
        ↓
Load ResNet50 Weights
        ↓
Set Model to Evaluation Mode
        ↓
Application Ready
```

---

# 🚀 End-to-End Cloud Deployment

The application is designed for cloud deployment using **Render**.

## Deployment Architecture

```text
                         GitHub
                           │
                           │ Source Code
                           ▼
                  ┌──────────────────┐
                  │      Render      │
                  │   Flask +        │
                  │    Gunicorn      │
                  └────────┬─────────┘
                           │
                           │ Download Model
                           ▼
                  ┌──────────────────┐
                  │  Hugging Face    │
                  │    Model Hub     │
                  └────────┬─────────┘
                           │
                           ▼
                 bt_resnet50_model.pt
                           │
                           ▼
                    ResNet50 Model
                           │
                           ▼
                    MRI Prediction
                           │
                           ▼
                     Live Web App
```

### GitHub

Contains:

```text
Source Code
Flask Application
Frontend Templates
Training Scripts
Requirements
Documentation
```

### Hugging Face

Contains:

```text
Trained ResNet50 Model
bt_resnet50_model.pt
```

### Render

Runs:

```text
Flask Application
        +
Gunicorn
        +
Downloaded ResNet50 Model
```

---

# ☁️ Render Deployment Configuration

The application can be deployed as a **Render Web Service**.

### Runtime

```text
Python 3
```

### Branch

```text
main
```

### Root Directory

```text
Blank
```

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

The application reads the `PORT` environment variable provided by the hosting platform:

```python
port = int(os.environ.get("PORT", 5000))
```

This allows the application to run both locally and on a cloud platform.

---

# 💻 Local Installation

## 1. Clone the Repository

```bash
git clone https://github.com/vickycodeswith/Brain-Tumor-Detection.git
```

Move into the project directory:

```bash
cd Brain-Tumor-Detection
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Flask Application

The trained model is automatically downloaded from Hugging Face when the application starts.

Run:

```bash
python app.py
```

The Flask application will start at:

```text
http://127.0.0.1:5000
```

If port 5000 is already occupied, use another port:

```bash
PORT=5001 python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

---

# 🧪 Running the Training Pipeline

If the original dataset is available locally, the complete training pipeline can be executed.

## Step 1 — Preprocess Dataset

```bash
python src/preprocess.py
```

This converts the original `.mat` files into processed MRI images.

## Step 2 — Create Patient-Level Splits

```bash
python src/split_dataset.py
```

This creates:

```text
train.csv
val.csv
test.csv
```

with patient-level separation.

## Step 3 — Train the Model

```bash
python src/train.py
```

The training process uses ResNet50 transfer learning.

---

# 📁 Project Structure

```text
Brain-Tumor-Detection/
│
├── Brain-Tumor-Test-Images/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
│
├── models/
│   └── README.md
│
├── src/
│   ├── preprocess.py
│   ├── split_dataset.py
│   └── train.py
│
├── static/
│   └── b.jpg
│
├── templates/
│   ├── Diseasedet.html
│   ├── MainPage.html
│   ├── error.html
│   ├── pred.html
│   └── uimg.html
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Programming language |
| **PyTorch** | Deep learning framework |
| **Torchvision** | Computer vision models and transforms |
| **ResNet50** | Deep learning architecture |
| **Flask** | Web application backend |
| **Gunicorn** | Production WSGI server |
| **Hugging Face Hub** | Trained model hosting |
| **Render** | Cloud deployment |
| **HTML/CSS** | Frontend |
| **Pillow** | Image processing |
| **NumPy** | Numerical computation |
| **Pandas** | Dataset management |
| **h5py** | MATLAB `.mat` file processing |
| **scikit-learn** | Dataset splitting and evaluation |

---

# 📦 Large Files and GitHub

The following files are intentionally excluded from this repository:

```text
dataset/
processed_dataset/
brain_tumor_training_data.tar.gz
metadata.csv
train.csv
val.csv
test.csv
models/bt_resnet50_model.pt
```

These files are excluded using `.gitignore`.

### Repository Design

```text
GitHub
│
├── Source Code
├── Flask Application
├── Training Scripts
├── Frontend
└── Documentation

Hugging Face
│
└── Trained ResNet50 Model
```

This keeps the GitHub repository lightweight while allowing the deployed application to access the trained model.

---

# 🔒 Data Privacy & Repository Design

The repository does not contain the complete training dataset.

Large generated and training files are excluded to:

- Keep repository size small
- Avoid unnecessary large-file storage
- Prevent accidental dataset uploads
- Separate training artifacts from source code
- Make the project easier to clone and review

The trained model is maintained separately on Hugging Face.

---

# 🔬 Machine Learning Workflow

The overall machine learning workflow is:

```text
1. Dataset Collection
        ↓
2. .mat File Processing
        ↓
3. Image Normalization
        ↓
4. Patient Identification
        ↓
5. Patient-Level Dataset Split
        ↓
6. Training / Validation / Test
        ↓
7. ResNet50 Transfer Learning
        ↓
8. Model Fine-Tuning
        ↓
9. Validation
        ↓
10. Test Evaluation
        ↓
11. Trained Model
        ↓
12. Hugging Face Model Hosting
        ↓
13. Flask Integration
        ↓
14. Render Deployment
        ↓
15. Live MRI Prediction
```

---

# 🎯 Key Results

```text
Dataset Images:          3,064
Unique Patients:           233

Train Images:            2,153
Validation Images:         457
Test Images:               454

Best Validation Accuracy: 95.84%
Test Accuracy:            94.05%

Model:                    ResNet50
Framework:                PyTorch
Training GPU:             Tesla T4
```

---

# 🧠 What This Project Demonstrates

This project demonstrates practical experience with:

- Deep Learning
- Computer Vision
- Transfer Learning
- CNN architectures
- ResNet50
- PyTorch
- Image preprocessing
- Dataset engineering
- Patient-level data splitting
- Model evaluation
- Classification metrics
- Confusion matrices
- Flask development
- Cloud deployment
- Model hosting
- Frontend development
- Git and GitHub
- Hugging Face Model Hub
- GPU-based model training

---

# ⚠️ Medical Disclaimer

**This project is for educational and research purposes only.**

This application is **not a certified medical diagnostic system** and should not be used to make medical decisions.

Predictions generated by this model should not replace:

- Professional medical evaluation
- Radiologist interpretation
- Clinical diagnosis
- Consultation with a qualified healthcare professional

Always consult an appropriate medical professional for real-world medical concerns.

---

# 👨‍💻 Developer

## Nitesh Yadav

GitHub:

https://github.com/vickycodeswith

Project Repository:

https://github.com/vickycodeswith/Brain-Tumor-Detection

Model Repository:

https://huggingface.co/Vicky8021/brain-tumor-resnet50

---

# ⭐ Acknowledgements

This project uses several open-source technologies and libraries, including:

- PyTorch
- Torchvision
- Flask
- Gunicorn
- Hugging Face Hub
- NumPy
- Pandas
- Pillow
- scikit-learn
- h5py

---

# 📜 License

This project is intended primarily for **educational and research purposes**.

---

## ⭐ Support the Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

**Built with Python, PyTorch, ResNet50, Hugging Face and Flask. 🧠🚀**

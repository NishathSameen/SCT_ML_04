# ✋ Hand Gesture Recognition System

A real-time Hand Gesture Recognition System developed using Python, TensorFlow, OpenCV, and Convolutional Neural Networks (CNN). The project is trained using the Kaggle LeapGestRecog dataset and predicts different hand gestures through live webcam input.

---

## 📌 Project Overview

Hand gesture recognition is a Computer Vision application that enables machines to understand human hand movements and gestures. This project uses Deep Learning techniques to classify hand gestures and perform real-time predictions using webcam input.

The system captures the hand region from the webcam, preprocesses the image, and sends it to a trained CNN model that predicts the gesture with a confidence score.

---

## 🚀 Features

✅ Real-time webcam gesture detection

✅ CNN-based image classification model

✅ Hand region extraction using OpenCV

✅ Supports multiple hand gestures

✅ Displays prediction confidence score

✅ Live visual recognition interface

✅ Trained on Kaggle LeapGestRecog dataset

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning Model |
| OpenCV | Image Processing & Webcam Integration |
| NumPy | Numerical Computation |
| Scikit-Learn | Data Splitting & Utilities |
| CNN | Gesture Classification |

---

## 📂 Dataset

Dataset used:

**LeapGestRecog Dataset (Kaggle)**

The dataset contains thousands of hand gesture images collected from multiple users and includes different gesture categories.

Supported gesture classes:

- Palm
- L
- Fist
- Fist Moved
- Thumb
- Index
- OK
- Palm Moved
- C
- Down

---

## ⚙ Project Workflow

### Step 1: Dataset Loading
- Load images from the LeapGestRecog dataset
- Assign labels to each gesture class

### Step 2: Image Preprocessing
- Resize images to 64×64 pixels
- Normalize pixel values
- Convert labels into categorical format

### Step 3: Model Training
- Build CNN architecture
- Add convolution and pooling layers
- Train model on gesture images

### Step 4: Model Evaluation
- Test model accuracy on unseen data
- Save trained model

### Step 5: Real-time Prediction
- Capture webcam input
- Detect hand region
- Predict gesture using trained model

---

## 📁 Project Structure

```text
Hand-Gesture-Recognition/
│
├── leapGestRecog/
│
├── hand_gesture.py
├── gesture_camera.py
├── hand_gesture_model.h5
├── gesture_labels.pkl
├── requirements.txt
└── README.md
```

## ▶ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/Hand-Gesture-Recognition.git
```

Move into project folder:

```bash
cd Hand-Gesture-Recognition
```

Install dependencies:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

---

## ▶ Running the Project

Train the model:

```bash
python hand_gesture.py
```

Run real-time webcam prediction:

```bash
python gesture_camera.py
```

Press:

```text
Q → Exit application
```

---

## 📊 Project Outcome

Successfully developed a real-time AI-based hand gesture recognition system capable of detecting and classifying multiple hand gestures with live webcam prediction.

---

## 🎯 Future Improvements

- Add more gesture classes
- Improve model accuracy
- Use MediaPipe for hand landmark detection
- Deploy as a web application
- Add gesture-controlled actions

SkillCraft Technology Internship Project

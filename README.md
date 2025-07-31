# SignLanguage-Classification

A lightweight, single-file Python project that classifies American Sign Language (ASL) alphabets (A–Z, excluding 'J' and 'Z') using a Convolutional Neural Network (CNN). Built with TensorFlow/Keras and trained on the Sign Language MNIST dataset.

---

## 🧠 Model Architecture

A Convolutional Neural Network (CNN) built from scratch with the following layers:

Input: 28x28 grayscale image

→ Conv2D (ReLU)  
→ MaxPooling2D  
→ Conv2D (ReLU)  
→ MaxPooling2D  
→ Flatten  
→ Dense (ReLU)  
→ Dropout  
→ Dense (Softmax)

Output: 24-class classification (A–Z, skipping 'J' and 'Z')

---

## 📦 Libraries Used

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  

import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

Install them using:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

📊 Dataset
Source: https://www.kaggle.com/datamunge/sign-language-mnist

Files:

sign_mnist_train.csv

sign_mnist_test.csv

Image Format: 28×28 grayscale (flattened into 784 columns)

Labels: Integers from 0 to 25 (excluding label 9 for ‘J’)

🚀 How to Run
Download the dataset and place the CSVs in your project folder.

Run the script:
python Sign Language Classification.ipynb

✅ Output Example
Sample predictions from the test set:

Predicted: C - Actual: C
Predicted: L - Actual: L
Predicted: V - Actual: V
Predicted: H - Actual: H
Predicted: A - Actual: A

The script also prints:

📉 Classification report (precision, recall, F1-score)

🔍 Confusion matrix as a heatmap (via Seaborn)

⚠️ Limitations
❌ Does not support dynamic signs like ‘J’ or ‘Z’, which require motion.

📏 Only works on 28×28 grayscale images — not adaptable to higher-res or color inputs.

🌥️ Sensitive to lighting and background noise — limited generalization.

🧪 Trained on static images only, not on real-world hand gestures.

🎥 No real-time webcam or video input implemented (yet).

🚀 Future Work & Improvements
📸 Integrate real-time webcam input using OpenCV or MediaPipe.

👐 Support dynamic signs using LSTM, 3D CNNs, or pose estimation.

🌍 Expand dataset to include diverse skin tones, hand shapes, and lighting conditions.

💾 Add model checkpointing, export to .h5, and build a reusable inference pipeline.

📱 Convert into a web app (TensorFlow.js) or mobile app (TensorFlow Lite) for accessibility.




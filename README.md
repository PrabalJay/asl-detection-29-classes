## 🖐️ ASL Detection System (29 Classes)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

*A deep learning-based American Sign Language (ASL) recognition system that classifies 29 ASL alphabet signs (A-Z, SPACE, DEL, NOTHING) from images or webcam in real time.*

---

## Table of Contents 📑
1. [Features](#-features)
2. [Demo](#-demo)
3. [Installation](#-installation)
4. [Requirements](#-requirements)
5. [Quick Start](#-quick-start)
6. [Usage](#-usage)
7. [Project Structure](#-project-structure)
8. [Training the Model](#-training-the-model)
9. [Evaluation](#-evaluation)
10. [Dataset](#-dataset)
11. [Contributing](#-contributing)
12. [Acknowledgements](#-acknowledgements)
13. [Live Demo](#-live-demo)
14. [License](#-license)
15. [Troubleshooting](#-troubleshooting)
16. [Credits](#-credits)

---

## ✨ Features

- 🔤 Recognizes all ASL alphabets (A-Z)
- ✋ Detects SPACE, DEL, and NOTHING gestures
- 🎥 Real-time webcam and image input support
- 📈 100% test accuracy on all classes
- 🖼️ Confusion matrix visualization
- 🛠️ Easy-to-use scripts for training, detection, and evaluation

---

## 🎥 Demo

![Preview](sample_1.png)
![Confusion Matrix](confusion_matrix.png)  

---

## ⚙️ Installation

Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for training)

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🏁 Quick Start

1. Clone the repository
```bash
git clone https://github.com/PrabalJay/asl-detection-29-classes.git
cd asl-detection-29-classes
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the model
```bash
python train.py
```

4. Evaluate the model
```bash
python evaluate.py
```

5. Run real-time detection
```bash
python detect.py
```

---

## 💡 Usage

- **Train the Model:**
  - `python train.py` — Trains the ASL recognition model using the dataset in `data/train/`.
- **Evaluate the Model:**
  - `python evaluate.py` — Evaluates the trained model and generates a confusion matrix.
- **Real-Time Detection:**
  - `python detect.py` — Launches webcam-based or image-based ASL detection.
- **Check Dataset Images:**
  - `python check_images.py` — Verifies and cleans dataset images.
- **Configuration:**
  - `python config.py` — Adjusts configuration settings for training and detection.

1. Detect ASL in an Image
bash
python detect.py 
Example output:
Predicted: Sign Y (1.00)

2. Real-time Webcam Detection
bash
python detect.py --webcam
(Press 'Q' to quit)

![Preview](sample_2.png)

---

## 📁 Project Structure

```
asl-recognition/
├── data/
│   ├── train/         # Training images
│   └── test/          # Test images
├── models/
│   └── model.keras    # Trained model
├── check_images.py    # Dataset verification utility
├── config.py          # Configuration file
├── detect.py          # Real-time/image detection script
├── evaluate.py        # Model evaluation script
├── train.py           # Model training script
├── requirements.txt   # Python dependencies
├── confusion_matrix.png # Model performance visualization
└── README.md
```

---

##🎯 Training the Model

Organize dataset as:

data/train/
  ├── A/     # Images for 'A'
  ├── B/
  └── ...
Run training:

bash
python train.py
Automatically saves best model

Generates confusion_matrix.png

![Confusion Matrix](confusion_matrix.png)

---

##📊 Evaluation

Get performance metrics:

bash
python evaluate.py
Sample output:

Test Accuracy: 96.3%
Class-wise Precision:
A: 0.98 | B: 0.95 | ...

---

##📂 Dataset

Source: ASL Alphabet Dataset on Kaggle

Requirements:

3,000+ images total

Balanced classes (~100 images/letter)

PNG/JPG format

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.

---

## 🙏 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) — Deep learning framework
- [OpenCV](https://opencv.org/) — Computer vision library
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — Dataset used for training and testing
- [scikit-learn](https://scikit-learn.org/) — Evaluation and metrics
- [Matplotlib](https://matplotlib.org/) — Visualization

Special thanks to the open-source community and contributors for their resources and support.

---

## Live Demo  
[Try it here](https://PrabalJay.github.io/asl-detection-29-classes)  

---

📜 License

MIT License - See LICENSE for details.

---

🛠️ Troubleshooting

Issue	Solution
CUDA errors	Install correct TensorFlow-GPU
Missing dependencies	Run pip install -r requirements.txt
Low accuracy	Increase epochs or add more training data

---

🙌 Credits

Built with TensorFlow/Keras

---

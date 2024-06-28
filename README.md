### Face Recognition System using OpenCV and Python
This project implements a real-time face recognition system using OpenCV and Python. 
It detects faces through the webcam, identifies them based on a trained model, and displays the results in real-time.

## Features
1. Face Detection: Utilizes the Haar Cascade classifier for detecting faces in the webcam stream.

2. Face Recognition: Implements LBPH (Local Binary Patterns Histograms) Face Recognizer from OpenCV for recognizing faces based on a trained dataset.

3. Real-time Display: Draws rectangles around detected faces and labels them with their IDs and confidence levels.

## Prerequisites
1. Python 3.x
2. OpenCV (cv2) library
3. PIL (Python Imaging Library) or Pillow
   
## Installation
Clone the repository:
git clone https://github.com/your-username/face-recognition.git
cd face-recognition

## Install dependencies:

pip install opencv-python-headless Pillow

## Usage
Train the face recognition model by running training.py:

python training.py
This script will create trainer/trainer.yml with the trained model data.

## Run the face recognition system:
python face_recognition.py
This will start the webcam and display real-time face recognition results.


## Directory Structure

face-recognition/
│
├── trainer/
│   └── trainer.yml      # Trained model file
│
├── haarcascade_frontalface_default.xml   # Haar Cascade classifier
│
├── face_recognition.py   # Main script for face recognition
├── training.py           # Script to train the face recognition model
│
└── dataset/              # Directory containing face images for training
    ├── User1/            # Subdirectory for user 1's images
    ├── User2/            # Subdirectory for user 2's images
    └── ...               # Additional subdirectories for other users


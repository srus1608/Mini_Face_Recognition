import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')  # Convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        except Exception as e:
            print(f"[ERROR] Could not process image {imagePath}: {e}")
            continue

    return face_samples, ids

print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

if len(faces) == 0 or len(ids) == 0:
    print("\n[ERROR] No faces found in the dataset directory or error in reading the files.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer/trainer.yml')
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program.")

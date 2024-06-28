import cv2
import os

# Ask for user input
user_id = input('Enter user_id and press Enter: ')

print("\n[INFO] Initializing face capture. Look at the camera and wait ...")

# Create a directory to save the dataset if it does not exist
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize the video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize individual sampling face count
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the dataset folder
        file_name = f"User.{user_id}.{count}.jpg"
        save_path = os.path.join(dataset_dir, file_name)
        print(f"[INFO] Saving image {count}: {save_path}")
        cv2.imwrite(save_path, gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' to exit
    if k == 27:
        break
    elif count >= 30:  # Take 30 face samples and stop the video
        break

# Clean up
print("\n[INFO] Exiting program and cleaning up.")
cam.release()
cv2.destroyAllWindows()

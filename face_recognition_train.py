import cv2
import os
import numpy as np

def train_recognizer(data_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = []
    ids = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = detector.detectMultiScale(gray)
                for (x, y, w, h) in faces_detected:
                    faces.append(gray[y:y+h, x:x+w])
                    ids.append(int(os.path.basename(root)))

    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')

    return faces, ids, recognizer

if __name__ == '__main__':
    train_recognizer('known_faces')

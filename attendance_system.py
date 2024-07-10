import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle

# Encode faces from a directory named "known_faces"
def encode_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            face_encoding = face_recognition.face_encodings(image)[0] 
            known_face_encodings.append(face_encoding)

            # Extract only the name without any numbers or extensions
            name = os.path.splitext(filename)[0] 
            name = name.split("(")[0].strip()  # Remove any numbers in parentheses
            known_face_names.append(name) 

    return known_face_encodings, known_face_names
# Load MobileNet model for face detection
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Function to mark attendance
def mark_attendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        date_time_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{name},{date_time_string}\n")
        print(f"Attendance marked for {name} at {date_time_string}")

# Load known face encodings and names (assuming you have a "known_faces" directory)
known_face_encodings, known_face_names = encode_known_faces("known_faces")

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the current frame using MobileNet
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Loop through detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Filter out weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Find face encodings for detected face
            face_encoding = face_recognition.face_encodings(rgb_small_frame, [(startY, endX, endY, startX)])[0]

            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                mark_attendance(name)

            # Draw the results
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
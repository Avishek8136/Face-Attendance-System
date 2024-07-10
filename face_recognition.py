import cv2
import dlib
import numpy as np
import os

# Load the pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Function to encode a face
def encode_face(image):
    """Encodes a face using dlib's landmark detector.

    Args:
        image: A NumPy array representing the image.

    Returns:
        A NumPy array representing the face encoding, or None if no face is detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        face_descriptor = np.array([landmarks.part(i).x for i in range(68)] + [landmarks.part(i).y for i in range(68)])
        return face_descriptor
    else:
        return None

# Function to compare two face encodings
def compare_faces(encoding1, encoding2):
    """Compares two face encodings using cosine similarity.

    Args:
        encoding1: The first face encoding.
        encoding2: The second face encoding.

    Returns:
        True if the similarity is above a threshold (0.5 in this case), False otherwise.
    """
    similarity = 1 - np.linalg.norm(encoding1 - encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
    return similarity > 0.5

# Load known face encodings
known_face_encodings = {}
known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        image = cv2.imread(os.path.join(known_faces_dir, filename))
        encoding = encode_face(image)
        if encoding is not None:
            known_face_encodings[name] = encoding

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Detect faces in the current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Process detected faces
    for face in faces:
        # Get landmarks
        landmarks = predictor(gray, face)
        
        # Extract face encoding
        encoding = np.array([landmarks.part(i).x for i in range(68)] + [landmarks.part(i).y for i in range(68)])

        # Compare with known face encodings
        for name, known_encoding in known_face_encodings.items():
            if compare_faces(encoding, known_encoding):
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                break  # Stop checking after a match is found

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'esc' key press
    if cv2.waitKey(1) & 0xFF == 27:  # 'esc' key code
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
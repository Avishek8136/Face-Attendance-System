# Face Recognition and Attendance System

This project demonstrates a face recognition system that can:

1. Detect faces in real-time using a webcam.
2. Recognize known faces by comparing them with pre-encoded face data.
3. Mark attendance for recognized individuals in a CSV file.

## Features

* **Face Detection** : Uses dlib, OpenCV, or MobileNet models to detect faces.
* **Face Recognition** : Compares detected faces with a database of known faces.
* **Attendance Tracking** : Logs the name and timestamp of recognized individuals into an attendance file.

## Prerequisites

1. **Python** (version 3.6 or higher)
2. **Required Libraries** :

* `opencv-python`
* `dlib`
* `face_recognition`
* `numpy`

1. **Pre-trained Models** :

* `shape_predictor_68_face_landmarks.dat` for dlib
* `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` for MobileNet

1. **Directory Structure** :

* A folder named `known_faces` containing images of known individuals. Each image file name should correspond to the person’s name.

## Setup

1. Clone or download this repository.
2. Install the required libraries using pip:
   ```
   pip install opencv-python dlib face_recognition numpy
   ```
3. Ensure the `shape_predictor_68_face_landmarks.dat` and MobileNet model files are in the project directory.
4. Add images of known individuals to the `known_faces` folder.

## How to Run

### Face Recognition and Attendance

1. Run the script:
   ```
   python main_face_recognition.py
   ```
2. The program will access your webcam, detect faces, and recognize known faces in real time.
3. If a face is recognized, the name and timestamp will be logged into `attendance.csv`.

### Train Recognizer

To train an LBPH face recognizer:

1. Organize images in folders under `known_faces`, with folder names as numeric IDs for individuals.
2. Run:
   ```
   python train_recognizer.py
   ```
3. The model will be saved as `trainer.yml`.

## Project Components

### 1. **Main Face Recognition Script**

* Detects faces in real-time using a webcam.
* Compares detected faces with known face encodings.
* Marks attendance for recognized individuals.

### 2. **Face Encoding Script**

* Encodes faces in the `known_faces` directory into feature vectors.
* Stores names and encodings for use in recognition.

### 3. **Face Recognizer Training Script**

* Uses LBPH algorithm to train a face recognizer on labeled images.

## Files and Folders

* `known_faces/`: Directory for images of known individuals.
* `attendance.csv`: Attendance log file.
* `shape_predictor_68_face_landmarks.dat`: Pre-trained dlib model for facial landmarks.
* `deploy.prototxt` & `res10_300x300_ssd_iter_140000.caffemodel`: MobileNet face detection model.

## How It Works

1. **Detection** : Detects faces using dlib or MobileNet.
2. **Encoding** : Extracts unique features of faces.
3. **Comparison** : Matches detected faces with pre-encoded known faces.
4. **Logging** : If a match is found, logs the name and time in `attendance.csv`.

## Key Points

* Ensure the images in `known_faces` are clear and represent the individual accurately.
* Adjust the confidence threshold in MobileNet or similarity threshold in dlib as needed.
* Press `q` to exit the video feed.

## Limitations

* Accuracy depends on the quality of the images and models.
* May not perform well in poor lighting or with partial occlusions.

## Future Enhancements

* Add support for multiple cameras.
* Improve the user interface for easier management of known faces.
* Implement real-time alerts for unrecognized faces.

## License

This project is open-source and available for educational purposes. Modify and use it as needed.

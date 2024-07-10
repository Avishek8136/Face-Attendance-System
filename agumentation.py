import cv2
import numpy as np
import random
import os

def data_augmentation(image, save_path):
  """Augments an image by performing random transformations.

  Args:
    image: A NumPy array representing the image.
    save_path: The path to save the augmented image.
  """
  # Define possible transformations
  transformations = [
    lambda img: cv2.flip(img, 1),  # Horizontal flip
    lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),  # Rotate 90 degrees clockwise
    lambda img: cv2.rotate(img, cv2.ROTATE_180),  # Rotate 180 degrees
    lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),  # Rotate 90 degrees counterclockwise
    lambda img: cv2.GaussianBlur(img, (5, 5), 0),  # Gaussian blur
    lambda img: cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), -0.5, 0),  # Brightness adjustment
    lambda img: cv2.addWeighted(img, 0.5, np.zeros(img.shape, img.dtype), 0.5, 0),  # Brightness adjustment
    # Add more transformations as needed
  ]

  # Randomly choose a transformation
  transformed_image = random.choice(transformations)(image)

  # Save the augmented image
  cv2.imwrite(save_path, transformed_image)

# Get input from the user
input_dir = input("Enter the directory containing the images: ")
num_augmented_images = int(input("Enter the number of augmented images per original image (at least 4): "))

# Ensure at least 4 augmented images
if num_augmented_images < 4:
    num_augmented_images = 4
    print("Number of augmented images set to 4 (minimum).")

# Loop through images in the input directory
for filename in os.listdir(input_dir):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    image_path = os.path.join(input_dir, filename)
    print(f"Processing image: {filename}")
    image = cv2.imread(image_path)
    if image is None:
      print(f"Error: Could not load image from '{image_path}'")
    else:
      # Generate multiple augmented images
      for i in range(num_augmented_images):
        augmented_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
        augmented_path = os.path.join(input_dir, augmented_filename)  # Save in the same directory
        data_augmentation(image, augmented_path)

print("Data augmentation complete!")
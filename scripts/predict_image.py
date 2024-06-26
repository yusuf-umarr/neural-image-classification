import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
animal_model = load_model('models/animal_model.keras', compile=False)

# Function to preprocess a new image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
    img_array = img_to_array(img)  # Convert image to a numerical array
    img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    return img_array

# Capture an image from the laptop camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    print("Camera opened successfully. Press 'q' to capture an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            captured_frame = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame if ret else None

# Capture an image
new_image = capture_image_from_camera()
if new_image is not None:
    # Preprocess the captured image
    new_image_preprocessed = preprocess_image(new_image)

    # Make a prediction
    prediction = animal_model.predict(new_image_preprocessed)

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)

    # Map the predicted class to the corresponding label
    label_map_reverse = {0: 'cat', 1: 'cock', 2: 'dog'}
    predicted_label = label_map_reverse[predicted_class[0]]

    print(f'The predicted label is: {predicted_label}')
else:
    print("No image captured.")

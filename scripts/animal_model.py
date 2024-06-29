import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Define image paths and labels
image_list = [
    'data/dog6.jpeg',
    'data/dog5.png',
    'data/dog4.jpeg',
    'data/cat1.jpg',
    'data/cat2.jpg',
    'data/cat3.jpeg',
    'data/cock1.jpg',
    'data/cock2.jpg',
    'data/cock3.jpg',
    'data/dog1.jpeg',
    'data/dog2.jpg',
    'data/dog3.jpeg',
    'data/cock6.jpeg',
    'data/cock5.jpg',
    'data/cock4.jpeg',
    'data/cat6.jpg',
    'data/cat5.jpg',
    'data/cat4.jpeg',
]


labels = ['dog', 'dog', 'dog', 'cat', 'cat', 'cat', 'cock', 'cock', 'cock', 'dog', 'dog', 'dog', 'cock', 'cock', 'cock', 'cat', 'cat', 'cat']
label_map = {'cat': 0, 'cock': 1, 'dog': 2}

# Load and preprocess images
def load_and_preprocess_image(path):
    img = load_img(path, target_size=(128, 128))  # Resize to 128x128 pixels
    img_array = img_to_array(img)  # Convert image to a numerical array
    img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
    return img_array

images = np.array([load_and_preprocess_image(img_path) for img_path in image_list])
labels = np.array([label_map[label] for label in labels])

# Convert labels to categorical format
labels = tf.keras.utils.to_categorical(labels, num_classes=3)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

predictions = model.predict(X_val)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_val, axis=1)

# Make predictions
print(f'True labels: {true_classes}')
print(f'Predicted labels: {predicted_classes}')

# Save the trained model
model.save('models/animal_model.keras')

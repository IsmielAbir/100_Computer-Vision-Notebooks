# Step 1: Import libraries and packages
import tensorflow as tf
import cv2
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load and preprocess the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Convert the labels to one-hot encoded vectors
num_classes = len(set(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Reshape the images to be 28x28 pixels with a single color channel
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Step 3: Creating the CNN model architecture
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Step 4: Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Training the model
history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))

# Step 6: Evaluating the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# Step 7: Making predictions
# Load a new image and preprocess it
new_image = cv2.imread('a.png', cv2.IMREAD_GRAYSCALE)
new_image = cv2.resize(new_image, (28, 28))
new_image = np.expand_dims(new_image, axis=2)
new_image = np.expand_dims(new_image, axis=0)
new_image = new_image.astype('float32') / 255.

# Make a prediction and print the result
prediction = model.predict(new_image)
print("Prediction:", np.argmax(prediction))
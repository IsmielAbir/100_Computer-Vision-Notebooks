# Step 1: Import libraries and packages
import tensorflow as tf
import numpy as np
import cv2
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# Step 2: Load and preprocess the data
data = tfds.load("emnist/letters", split="train+test")
data = data.shuffle(buffer_size=1024).batch(32)
num_classes = 26

def preprocess_data(sample):
    image = sample['image']
    label = sample['label']
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label

data = data.map(preprocess_data)

# Split the data into training and testing sets
num_samples = data.cardinality().numpy()
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_data = data.take(train_size)
test_data = data.skip(train_size)

# Step 3: Creating the CNN model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Step 4: Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Training the model
history = model.fit(train_data, epochs=3, validation_data=test_data)

# Step 6: Evaluating the model
test_loss, test_acc = model.evaluate(test_data, verbose=0)
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
predicted_char = chr(np.argmax(prediction) + 65)
print("Prediction:", predicted_char)

# Display the image and predicted character
plt.imshow(new_image[0,:,:,0], cmap='gray')
plt.title("Predicted character: " + predicted_char)
plt.show()
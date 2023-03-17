import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')

'''
#for x in range(1,3):
img = cv.imread('two.png')[:,:,0]
    #np.np.invert(np.array([img]))
    #prediction = model.predict(img)
img = tf.keras.utils.normalize(img, axis=1)
prediction = model.predict(img.reshape(1, 28, 28))
#prediction = model.predict(img)
print(f'Result is: {np.argmax(prediction)}')
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
'''

#for x in range(1,3):
img = cv.imread('fo.png')[:,:,0]
img = cv.resize(img, (28,28)) # Resizing image to (28, 28)
img = tf.keras.utils.normalize(img, axis=1)
prediction = model.predict(img.reshape(1, 28, 28))
print(f'Result is: {np.argmax(prediction)}')
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
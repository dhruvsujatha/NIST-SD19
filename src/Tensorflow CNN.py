# %tensorflow_version 2.x  # this line is not required unless you are in a notebook
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.datasets import mnist
import numpy as np
from PIL import Image as im

(train_images, train_y), (test_images, test_y) = mnist.load_data()
np.reshape(train_images, (60000, 28, 28))
np.reshape(test_images, (10000, 28, 28))

print(np.shape(train_images), np.shape(train_y))
print(np.shape(test_images), np.shape(test_y))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_y, epochs=4,
                    validation_data=(test_images, test_y))

test_loss, test_acc = model.evaluate(test_images,  test_y, verbose=2)
print(str(test_acc * 100) + "  %")

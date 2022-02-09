import keras
from PIL import Image as im
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os

if __name__ == '__main__':
    images = []
    inPath = "C:/NN/NIST SD 19/DataSet/"
    i = 0
    for imagePath in os.listdir(inPath):
        inputPath = os.path.join(inPath, imagePath)
        img = im.open(inputPath)
        img = img.convert('1')
        pixels = np.asarray(img)
        images.append(pixels)
        img.close()

    images_array = np.reshape(images, (31000, 2500))
    images_array = np.reshape(images, (62, 500, 2500))

    training_data = images_array[:, 0:400, :]
    training_data = np.reshape(training_data, (24800, 2500))
    training_data = training_data / 255

    test_data = images_array[:, 400:500, :]
    test_data = np.reshape(test_data, (6200, 2500))
    test_data = test_data / 255

    training_labels = np.empty(0)
    test_labels = np.empty(0)
    for i in range(62):
        to_add = i * np.ones(400)
        training_labels = np.append(training_labels, to_add)
        to_add = i * np.ones(100)
        test_labels = np.append(test_labels, to_add)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(50, 50)),
        keras.layers.Dense(750, activation='relu'),
        keras.layers.Dense(62, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_data, training_labels, epochs=1)


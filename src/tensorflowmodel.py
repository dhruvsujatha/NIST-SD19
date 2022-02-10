from PIL import Image as im
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import datasets, layers, Sequential, models
import numpy as np

train_data_points = 400
test_data_points = 100
total_data_points = train_data_points + test_data_points
pixels = 50
bins = 62
mid_layer = 750
training_iters = 10
learning_rate = 0.001
batch_size = 128

images = []
inPath = "C:/NN/NIST SD 19/DataSet/"
i = 0
for i in range(48, 123):
    p = np.round(((i - 48) / 75) * 100, 2)
    print('\r''Data loading:', p, '%', end='')
    for j in range(total_data_points):
        if 47 < i < 58 or 64 < i < 91 or 96 < i < 123:
            input_path = inPath + str(i) + '-' + str(j) + '.png'
            img = im.open(input_path)
            img = img.convert('1')
            img = img.resize((pixels, pixels))
            img_array = np.asarray(img)
            images.append(img_array)
            img.close()

images_array = np.reshape(images, (bins, total_data_points, pixels, pixels))

training_data = images_array[:, 0:train_data_points, :, :]
training_data = np.reshape(training_data, (train_data_points * bins, pixels, pixels, 1))

test_data = images_array[:, train_data_points:total_data_points, :, :]
test_data = np.reshape(test_data, (test_data_points * bins, pixels, pixels, 1))

training_labels = np.zeros((train_data_points, bins))
test_labels = np.zeros((test_data_points, bins))
for i in range(train_data_points):
    training_labels[i][(i // train_data_points)] = 1
for i in range(test_data_points):
    test_labels[i][(i // test_data_points)] = 1

training_data = training_data / 255.0
test_data = test_data / 255.0

x = tf.compat.v1.placeholder("float", [None, pixels, pixels, 1])
y = tf.compat.v1.placeholder("float", [None, bins])

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


weights = {
    'wc1': tf.compat.v1.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.initializers.GlorotUniform()),
    'wc2': tf.compat.v1.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.initializers.GlorotUniform()),
    'wc3': tf.compat.v1.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.initializers.GlorotUniform()),
    'wd1': tf.compat.v1.get_variable('W3', shape=(4*4*128, 128), initializer=tf.initializers.GlorotUniform()),
    'out': tf.compat.v1.get_variable('W6', shape=(128, bins), initializer=tf.initializers.GlorotUniform()),
}
biases = {
    'bc1': tf.compat.v1.get_variable('B0', shape=(32), initializer=tf.initializers.GlorotUniform()),
    'bc2': tf.compat.v1.get_variable('B1', shape=(64), initializer=tf.initializers.GlorotUniform()),
    'bc3': tf.compat.v1.get_variable('B2', shape=(128), initializer=tf.initializers.GlorotUniform()),
    'bd1': tf.compat.v1.get_variable('B3', shape=(128), initializer=tf.initializers.GlorotUniform()),
    'out': tf.compat.v1.get_variable('B4', shape=(bins), initializer=tf.initializers.GlorotUniform()),
}


def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

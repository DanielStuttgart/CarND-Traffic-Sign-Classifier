### Load data #############################################################################################
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import os

# change working dir to path containing this file
os.chdir(os.path.dirname('C:/Users/P325748/Documents/4_AI/Udacity/CarND-Traffic-Sign-Classifier-Project-master/'))

# traffic-signs-data
training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

import numpy as np

# Pad images with 0s --> not necessary for traffic signs
#X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
#X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
#X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Error in video @0:35 s: validation set is already given. 
# However, here is the code to split training-data into training and validation
#from sklearn.model_selection import train_test_split
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print("Updated Image Shape: {}".format(X_train[0].shape))

### Preprocess data #############################################################################################
# preprocess s.t. mean of data = 0 and equal variance --> img_data == 0..255 --> img_data - 128
#X_train = (X_train.astype(float) - 128) * 128
#X_valid = (X_valid.astype(float) - 128) * 128
#X_test = (X_test.astype(float) - 128) * 128
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Visualize histogram
def hist(x, bins=50):
    hist, bins = np.histogram(x, bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    #plt.show()

def normalize(x):
    # per-channel standardization of pixels (see https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)
    means = x.mean(axis=(0,1), dtype='float64')
    stds = x.std(axis=(0,1), dtype='float64')
    x = (x - means) / stds
    x = np.clip(x, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    x = (x + 1.0) / 2.0

    return x

# idea taken from https://medium.com/@wolfapple/traffic-sign-recognition-2b0c3835e104
def normalize_hist(x):
    # convert to Y, Cr, Cb --> only Y necessary for further evaluation (in HLS, L was best); (in HSV, S was good)
    #img_y = cv2.cvtColor(x.reshape(-1,32,3), cv2.COLOR_RGB2HSV)[:,:,1]
    #img_y = cv2.cvtColor(x.reshape(-1,32,3), cv2.COLOR_RGB2GRAY)
    img_y = cv2.cvtColor(x.reshape(-1,32,3), cv2.COLOR_RGB2YCrCb)[:,:,0]
    # do histogram (not on whole image equally but on parts with contrast limiting)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    img_clahe = clahe.apply(img_y)
    x = img_clahe.reshape(int(img_clahe.shape[0]/32), 32, 32, 1)

    # try to center pixel values
    x = (x - 128.0) / 128

    return x

def normalize_udacity(x):
    # as proposed in udacity tutorial
    x = (x - 128.0) / 128

    return x

X_train_old = X_train   # store old values for visualization
#X_train = normalize_udacity(X_train)    # for RGB
#X_test = normalize_udacity(X_test)
#X_valid = normalize_udacity(X_valid)
X_train = normalize_hist(X_train)      # for Gray
X_test = normalize_hist(X_test)
X_valid = normalize_hist(X_valid)

### Visualize data #############################################################################################
#%matplotlib inline
# for visualization of dataset, please see "dataset_visualization.py"

index = random.randint(0, len(X_train))
#image = X_train[index]              # for RGB-images
image = X_train[index].squeeze()   # for gray scale

plt.figure()
plt.title('Image has following class: ' + str(y_train[index]))
plt.subplot(2,2,1)
plt.imshow(X_train_old[index])
plt.subplot(2,2,2)
hist(X_train_old[index])
plt.subplot(2,2,3)
plt.imshow(image, cmap='gray')
plt.subplot(2,2,4)
hist(image)
plt.show()


### Preprocess data #############################################################################################
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

### Prepare Tensorflow #############################################################################################
import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 128        # [128]

### Definition of Architecture #############################################################################################
from tensorflow.contrib.layers import flatten

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0          # important to set values 
sigma = 0.1     # standard deviation == 1.0, if nothin is given --> huge difference

weights = {
    #'1_conv_layer': tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean = mu, stddev = sigma), name='1_conv_weight'), # for RGB
    '1_conv_layer': tf.Variable(tf.truncated_normal(shape=(5,5,1,12), mean = mu, stddev = sigma), name='1_conv_weight'), # for Gray
    '2_conv_layer': tf.Variable(tf.truncated_normal(shape=(5,5,12,16), mean = mu, stddev = sigma), name='2_conv_weight'),
    '3_mul_layer': tf.Variable(tf.truncated_normal(shape=(400,120), mean = mu, stddev = sigma), name='3_mul_weight'),
    '4_mul_layer': tf.Variable(tf.truncated_normal(shape=(120,84), mean = mu, stddev = sigma), name='4_mul_weight'), 
    '5_mul_layer': tf.Variable(tf.truncated_normal(shape=(84,43), mean = mu, stddev = sigma), name='5_mul_weight')
}
biases = {
    '1_conv_layer': tf.Variable(tf.zeros([12]), name='1_conv_bias'),
    '2_conv_layer': tf.Variable(tf.zeros([16]), name='2_conv_bias'),
    '3_mul_layer': tf.Variable(tf.zeros([120]), name='3_mul_bias'),
    '4_mul_layer': tf.Variable(tf.zeros([84]), name='4_mul_bias'),
    '5_mul_layer': tf.Variable(tf.zeros([43]), name='5_mul_bias')
}  

def LeNet(x):    
  
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x12.
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    conv_layer1 = tf.nn.conv2d(x, weights['1_conv_layer'], strides=[1, 1, 1, 1], padding='VALID') + biases['1_conv_layer']

    # TODO: Activation.
    conv_layer1 = tf.nn.relu(conv_layer1)

    # new: dropout; usually applied after activation; when using ReLU, dropout even before possible
    conv_layer1 = tf.nn.dropout(conv_layer1, rate=0.25)

    # TODO: Pooling. Input = 28x28x12. Output = 14x14x12.
    pooling1 = tf.nn.max_pool(conv_layer1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    # new: dropout
    #pooling1 = tf.nn.dropout(pooling1, rate=0.5)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv_layer2 = tf.nn.conv2d(pooling1, weights['2_conv_layer'], strides=[1, 1, 1, 1], padding='VALID') + biases['2_conv_layer']
    
    # TODO: Activation.
    conv_layer2 = tf.nn.relu(conv_layer2)

    # new: dropout; usually applied after activation; when using ReLU, dropout even before possible
    #conv_layer2 = tf.nn.dropout(conv_layer2, rate=0.5)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pooling2 = tf.nn.max_pool(conv_layer2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    # new: dropout
    #pooling2 = tf.nn.dropout(pooling2, rate=0.75)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat1 = tf.contrib.layers.flatten(pooling2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # in order to reduce dimensionality in fully connected layer, matrix multipliction is used
    layer3 = tf.matmul(flat1, weights['3_mul_layer']) + biases['3_mul_layer']
    
    # TODO: Activation.
    layer3 = tf.nn.relu(layer3)

    # new: dropout; usually applied after activation; when using ReLU, dropout even before possible
    #layer3 = tf.nn.dropout(layer3, rate=0.5)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    layer4 = tf.matmul(layer3, weights['4_mul_layer']) + biases['4_mul_layer']

    # TODO: Activation.
    layer4 = tf.nn.relu(layer4)

    # new: dropout
    #layer4 = tf.nn.dropout(layer4, rate=0.2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43. (43 traffic sign classes)
    # --> 43 == number of output classes [0..42]
    logits = tf.matmul(layer4, weights['5_mul_layer']) + biases['5_mul_layer']

    # new: dropout
    #logits = tf.nn.dropout(logits, rate=0.5)

    return logits    

### Features x and labels y #############################################################################################
#x = tf.placeholder(tf.float32, (None, 32, 32, 3)) # for RGB image
x = tf.placeholder(tf.float32, (None, 32, 32, 1)) # for gray
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

### Training pipeline #############################################################################################
rate = 0.005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
#loss_operation = tf.reduce_mean(cross_entropy)
loss_operation = tf.reduce_mean(cross_entropy) + 0.01*tf.nn.l2_loss(weights['1_conv_layer']) + 0.01*tf.nn.l2_loss(weights['2_conv_layer']) \
    + 0.01*tf.nn.l2_loss(weights['3_mul_layer']) + 0.01*tf.nn.l2_loss(weights['4_mul_layer']) + 0.01*tf.nn.l2_loss(weights['5_mul_layer'])
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

### Model evaluation #############################################################################################
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Train Model #############################################################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    validation_acc = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        validation_acc.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

### Evaluate model #############################################################################################
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
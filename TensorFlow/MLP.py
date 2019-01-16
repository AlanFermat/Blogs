import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran


def train_test_data(train_num, test_num):
	x_train = mnist.train.images[:train_num,:]
	y_train = mnist.train.labels[:train_num,:]
	x_test = mnist.test.images[:test_num,:]
	y_test = mnist.test.labels[:test_num,:]
	return x_train, y_train, x_test, y_test

def display_digit(num):
	print(y_train[num])
	label = y_train[num].argmax(axis=0)
	image = x_train[num].reshape([28,28])
	plt.title('Example: %d  Label: %d' % (num, label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()

dimension = 784
numClass = 10
x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

hiddenUnits = 28
W = tf.Variable(np.random.normal(0, 0.05, (dimension, hiddenUnits)), dtype=tf.float32)
b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)
W2 = tf.Variable(np.random.normal (0, 0.05, (hiddenUnits, numClass)),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, numClass)), dtype=tf.float32)

output0 = tf.tanh(tf.matmul(x_, W) + b) 
output1 = tf.matmul(output0, W2) + b2
y = tf.nn.softmax(output1)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
trainingAlg = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def display_compare(num, i):
    x_train = mnist.train.images[num,:].reshape(1,784)
    y_train = mnist.train.labels[num,:]
    label = y_train.argmax()
    prediction = sess.run(y, feed_dict={x_: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.savefig("pic " + str(i) + ".png")
    plt.show()


train_step = 1000
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	x_train, y_train, x_test, y_test = train_test_data(55000, 10000)
	for i in range(train_step+1):
		sess.run(trainingAlg, feed_dict={x_: x_train, y_: y_train})
		if i%100 == 0:
			display_compare(ran.randint(0, 55000), i)
			print('Training Step:' + str(i) + '  Accuracy =  ' 
				+ str(sess.run(accuracy, feed_dict={x_: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(loss, {x_: x_train, y_: y_train})))





	



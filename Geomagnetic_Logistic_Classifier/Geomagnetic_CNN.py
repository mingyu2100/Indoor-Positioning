import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

np.random.seed(777)
tf.set_random_seed(777)

xy_DB = np.loadtxt('Geomagnetic_DB_1m.csv', delimiter=',', dtype=np.float32)
xy_test = np.loadtxt('Geomagnetic_test_1m.csv', delimiter=',', dtype=np.float32)

db_classes = 90
read_num = 2
geo_m_db = xy_DB[:, [0]]
geo_x_db = xy_DB[:, [1]]
geo_y_db = xy_DB[:, [2]]
geo_z_db = xy_DB[:, [3]]

geo_m_test = xy_test[:, [0]]
geo_x_test = xy_test[:, [1]]
geo_y_test = xy_test[:, [2]]
geo_z_test = xy_test[:, [3]]

x_data = xy_DB[:, 0:-1]
y_data = xy_DB[:, [-1]]
# print(x_data.shape, y_data.shape)
# print(x_data, y_data)

x_test = xy_test[:, 0:-1]
y_test = xy_test[:, [-1]]
# print(x_test.shape, x_test.shape)
# print(x_test, x_test)

db_classes = 90
test_classes = 90
learning_rate = 0.01
training_epochs = 10
# batch_size = 100

class CNN_Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
    def _build_net(self):

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, db_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, db_classes])

W = tf.Variable(tf.random_normal([4, db_classes]), name='weight')
# W = tf.get_variable('W', shape=[4, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([db_classes]), name='bias')

logits = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for epoch in range(training_epochs):
for epoch in range(0):
    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

        if step % 2000 == 0:
            cost_val, acc_val = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print('epoch: %2d' % epoch, 'Step: %5d' % step, 'cost: ', cost_val, 'acc: ', acc_val)

pred = sess.run(prediction, feed_dict={X: x_data})
for p, y in zip(pred, y_data.flatten()):
    print('train_prediction: ', int(p), 'True: ', int(y))

# test
pred = sess.run(prediction, feed_dict={X: x_test})
for p, y in zip(pred, y_data.flatten()):
    print('test_prediction: ', int(p), 'True: ', int(y))
acc_val = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
print('acc: ', acc_val)

length_db = range(db_classes)
length_test = range(test_classes)
plt.figure()
plt.plot(length_db, geo_x_db, 'o-', label='x_db')
plt.plot(length_db, geo_y_db, 'o-', label='y_db')
plt.plot(length_db, geo_z_db, 'o-', label='z_db')
plt.plot(length_db, geo_m_db, 'o-', label='m_db')

plt.plot(length_test, geo_x_test, 's--', label='x_test')
plt.plot(length_test, geo_y_test, 's--', label='y_test')
plt.plot(length_test, geo_z_test, 's--', label='z_test')
plt.plot(length_test, geo_m_test, 's--', label='m_test')

plt.grid()
plt.legend()
plt.xlabel('meter')
plt.ylabel('value')
plt.title('Geomagnetic')
plt.show()
sess.close()
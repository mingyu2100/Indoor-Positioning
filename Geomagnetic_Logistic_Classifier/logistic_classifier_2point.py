# 1m단위(0m~1m, 1m~2m, ...) 방식으로 8point 비교하기

import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

np.random.seed(777)
tf.set_random_seed(777)

# files = glob.glob('*')
# print(files)

xy_DB = np.loadtxt('Geomagnetic_DB_1m.csv', delimiter=',', dtype=np.float32)
# xy_DB = np.loadtxt('Geomagnetic_test_1m_2.csv', delimiter=',', dtype=np.float32)
xy_test = np.loadtxt('Geomagnetic_test_1m.csv', delimiter=',', dtype=np.float32)

db_classes = 90
read_num = 2
test_classes = 90
learning_rate = 0.01
training_epochs = 10
# batch_size = 100

x_data = []
y_data = []
for i in range(0, db_classes-read_num+1):
    x_data.append(xy_DB[i: i+read_num, 0:-1])
    y_data.append(i)
x_data = np.array(x_data).reshape([-1, 4 * read_num])
y_data = np.array(y_data).reshape([-1, 1])
print(x_data, y_data)
print(x_data.shape, y_data.shape)

x_test = []
y_test = []
for i in range(0, test_classes-read_num+1):
    x_test.append(xy_test[i: i+read_num, 0:-1])
    y_test.append(i)
x_test = np.array(x_test).reshape([-1, 4 * read_num])
y_test = np.array(y_test).reshape([-1, 1])
# print(x_test, y_test)
# print(x_test.shape, y_test.shape)

X = tf.placeholder(tf.float32, [None, 4 * read_num])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, db_classes-read_num+1)
Y_one_hot = tf.reshape(Y_one_hot, [-1, db_classes-read_num+1])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([4 * read_num, db_classes-read_num+1]), name='weight1')
    # W = tf.get_variable('W', shape=[4, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([db_classes-read_num+1]), name='bias1')
    layer1 = tf.matmul(X, W1) + b1
    # layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    # layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([db_classes-read_num+1, db_classes-read_num+1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([db_classes-read_num+1]), name='bias2')
    logits = tf.matmul(layer1, W2) + b2

with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy') as scope:
    prediction = tf.argmax(logits, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
# merged_summary = tf.summary.merge_all()
# writer = tf.summary.FileWriter('./logs', sess.graph)
sess.run(tf.global_variables_initializer())

t_start = datetime.datetime.now()

for epoch in range(training_epochs):
# for epoch in range(4):
    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        # summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_data, Y: y_data})
        # writer.add_summary(summary, global_step=step)

        if step % 2000 == 0:
            cost_val, acc_val = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print('epoch: %2d' % epoch, 'Step: %5d' % step, 'cost: ', cost_val, 'acc: ', acc_val)

t_end = datetime.datetime.now()
print('Computation time: ' + str(t_end - t_start))

pred = sess.run(prediction, feed_dict={X: x_data})
for p, y in zip(pred, y_data.flatten()):
    print('train_prediction: ', int(p), 'True: ', int(y))

# test
pred = sess.run(prediction, feed_dict={X: x_test})
for p, y in zip(pred, y_data.flatten()):
    print('test_prediction: ', int(p), 'True: ', int(y))
acc_val = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
print('acc: ', acc_val)

sess.close()
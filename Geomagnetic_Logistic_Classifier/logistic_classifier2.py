# 다중 파일 읽어오기

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

read_num = 1

directory = os.listdir('./db')
# print(directory)

# xy_DB = []
# for file in directory:

filename_queue = tf.train.string_input_producer(['Geomagnetic_DB_1m.csv', 'Geomagnetic_test_1m.csv'], shuffle=False, name='filename_queue')
xy_test = np.loadtxt('Geomagnetic_test_1m_2.csv', delimiter=',', dtype=np.float32)

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

record_defaults = [[0.]] * 5
xy_DB = tf.decode_csv(value, record_defaults=record_defaults)
# train_x_batch, train_y_batch = tf.train.batch([xy_DB[0:-1], xy_DB[-1:]], batch_size=100)
train_x_batch, train_y_batch = tf.train.shuffle_batch([xy_DB[0:-1], xy_DB[-1:]], batch_size=100, capacity=50000, min_after_dequeue=10000)

# geo_m_db = xy_DB[:, [0]]
# geo_x_db = xy_DB[:, [1]]
# geo_y_db = xy_DB[:, [2]]
# geo_z_db = xy_DB[:, [3]]
#
# geo_m_test = xy_test[:, [0]]
# geo_x_test = xy_test[:, [1]]
# geo_y_test = xy_test[:, [2]]
# geo_z_test = xy_test[:, [3]]

x_test = xy_test[:, 0:-1]
y_test = xy_test[:, [-1]]
# print(x_test.shape, x_test.shape)
# print(x_test, x_test)

db_classes = 90
test_classes = 90
learning_rate = 0.01
training_epochs = 10

X = tf.placeholder(tf.float32, [None, 4], name='x-input')
Y = tf.placeholder(tf.int32, [None, 1], name='y-input')
Y_one_hot = tf.one_hot(Y, db_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, db_classes])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([4, 8]), name='weight1')
    # W = tf.get_variable('W', shape=[4, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([8]), name='bias1')
    layer1 = tf.matmul(X, W1) + b1
    # layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    # layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

    # W1_hist = tf.summary.histogram('weights1', W1)
    # b1_hist = tf.summary.histogram('biases1', b1)
    # layer1_hist = tf.summary.histogram('layers1', layer1)

with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([8, db_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([db_classes]), name='bias2')
    logits = tf.matmul(layer1, W2) + b2

#     # W2_hist = tf.summary.histogram('weights2', W2)
#     # b2_hist = tf.summary.histogram('biases2', b2)
#     # layer2_hist = tf.summary.histogram('layers2', layer2)

# with tf.name_scope('layer3') as scope:
#     W3 = tf.Variable(tf.random_normal([8, db_classes]), name='weight3')
#     b3 = tf.Variable(tf.random_normal([db_classes]), name='bias3')
#     logits = tf.matmul(layer2, W3) + b3

#     # W3_hist = tf.summary.histogram('weights3', W3)
#     # b3_hist = tf.summary.histogram('biases3', b3)
#     # layer3_hist = tf.summary.histogram('layers3', logits)

with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

with tf.name_scope('accuracy') as scope:
    prediction = tf.argmax(logits, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summ = tf.summary.scalar('accuracy', accuracy)

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)  # '/logs': C 드라이브에 생성, './logs': 프로젝트에 생성
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

t_start = datetime.datetime.now()

for epoch in range(training_epochs):
# for epoch in range(0):
    for step in range(10001):
        summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_batch, Y: y_batch})
        writer.add_summary(summary, global_step=step)

        if step % 2000 == 0:
            cost_val, acc_val = sess.run([cost, accuracy], feed_dict={X: x_batch, Y: y_batch})
            print('epoch: %2d' % epoch, 'Step: %5d' % step, 'cost: ', cost_val, 'acc: ', acc_val)

coord.request_stop()
coord.join(threads)

t_end = datetime.datetime.now()
print('Computation time: ' + str(t_end - t_start))

pred = sess.run(prediction, feed_dict={X: x_batch})
for p, y in zip(pred, y_batch.flatten()):
    print('train_prediction: ', int(p), 'True: ', int(y))

# test
pred = sess.run(prediction, feed_dict={X: x_test})
for p, y in zip(pred, y_test.flatten()):
    print('test_prediction: ', int(p), 'True: ', int(y))
acc_val = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
print('acc: ', acc_val)

# length_db = range(db_classes)
# length_test = range(test_classes)

# plt.figure()
# plt.plot(length_db, geo_m_db, 'ro-', label='m_db')
# plt.plot(length_db, geo_x_db, 'ro-', label='x_db')
# plt.plot(length_db, geo_y_db, 'ro-', label='y_db')
# plt.plot(length_db, geo_z_db, 'ro-', label='z_db')
#
# plt.plot(length_test, geo_m_test, 'bs--', label='m_test')
# plt.plot(length_test, geo_x_test, 'bs--', label='x_test')
# plt.plot(length_test, geo_y_test, 'bs--', label='y_test')
# plt.plot(length_test, geo_z_test, 'bs--', label='z_test')
#
# plt.grid()
# plt.legend()
# plt.xlabel('meter')
# plt.ylabel('value')
# plt.title('Geomagnetic')
# plt.show()
sess.close()

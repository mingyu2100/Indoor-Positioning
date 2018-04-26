import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

xy = np.loadtxt('accel.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# x_data = np.array(x_data, dtype=np.float32)
# y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(cost)
# optimizer = tf.train.AdamOptimizer(0.01).minimize(cost) # trash
# optimizer = tf.train.FtrlOptimizer(0.01).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

for epoch in range(30):
    for step in range(10000):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, optimizer], feed_dict={X: x_data, Y: y_data})

        if step % 2000 == 0:
            print('epoch: %2d' %epoch, 'step: %5d' % step, 'cost: ', cost_val, '\nW: ', W_val, 'b: ', b_val)

print('\n', epoch, step, cost_val, W_val, b_val)
# sess.run(hypothesis, feed_dict={X: })
sess.close()
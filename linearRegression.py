# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:41:09 2019

@author: Michael
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

learningRate = 0.001
epochs = 2000

nSamples = 30
trainX = np.linspace(0,20, nSamples)
trainY = 7 * trainX + 4 * np.random.randn(nSamples)


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = 'weights')
B = tf.Variable(np.random.randn(), name = 'bias')

pred = tf.add(tf.multiply(X, W), B)

cost = tf.reduce_sum(((pred - Y) ** 2) / (2 * nSamples))

optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        for x, y in zip(trainX, trainY):
            sess.run(optimizer, feed_dict= {X: x, Y: y})
            
        if not epoch % 20 or epoch == 0:
            c = sess.run(cost, feed_dict = {X: trainX, Y: trainY})
            w = sess.run(W)
            b = sess.run(B)
            print(f'epoch: {epoch:04d} c={c:.4f} w={w:.4f} b={b:.4f}')

    weight = sess.run(W)
    bias = sess.run(B)
    plt.plot(trainX, trainY, 'o')
    plt.plot(trainX, weight * trainX + bias)
    plt.show()

            
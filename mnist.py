#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:15:36 2018

@author: dmitriy
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from keras.optimizers import Adam

from keras.models import Model, Sequential
from keras.layers import Input, Dense

from keras import backend as K

def get_model():

        model = Sequential()
        model.add(Dense(324, input_dim=784, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(10, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam())
        return model


model = get_model()


for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  model.fit(batch_xs, batch_ys)


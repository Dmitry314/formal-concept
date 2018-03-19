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
from keras.models import load_model


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
  
  
get_2nd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])

outputs = []
answers = []
for i in range(1000):
    batch_x, answer =  mnist.test.next_batch(1)
    outputs.append(get_2nd_layer_output((batch_x)))
    answers.append(answer)
    
    
weights = []
for i in outputs:
    for j in i[0][0]:
        if(j!= 0):
            weights.append(j)
            
          
import pandas as pd
w = pd.DataFrame(weights)
    

import numpy as np
b = np.zeros([1000, 450])

for i in range(0, len(outputs)):
    for j in range(0, len(outputs[i][0][0])):
        if(outputs[i][0][0][j] < 3):
            b[i][2*j] = 1
        if(outputs[i][0][0][j] > 3 and  outputs[i][0][0][j] < 6):
            b[i][2*j + 1] = 1
        if(outputs[i][0][0][j] > 6):
            b[i][2*j+2] = 1

from fca import FCA, Diagram
import numpy as np
from data_loader import *
import time
from nn_layers import DeepSparseNet, DeepDenseNet
from lasagne import nonlinearities

X_train = b

fca = FCA(X_train)
l = fca.calculate_disjunctive_lattice()
# l = fca.calculate_lattice(num_level=4)
fca.save_lattice()
l = fca.load_lattice()

diag = Diagram(lattice=l, num_attributes=X_train.shape[1], X=b)
diag.calculate_child_concepts()
diag.save_child_concepts()
diag.load_child_concepts()


print(diag.child_concepts)


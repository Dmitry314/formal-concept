#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:06:17 2018

@author: dmitriy
"""

from fca import FCA, Diagram
import numpy as np
from data_loader import *
import time



import networkx as nx

from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.models import Model
from keras.layers import Input, Dense
X, y, object_labels, attribute_labels = get_titanic(file_name='data_sets//titanic.txt')[:4]
# get_random(1000, 15, frequency=0.8)[:4]
# get_titanic(file_name='data_sets//titanic.txt')[:4]
# get_mammographic_masses()[:4]
# get_breast_cancer()[:4]
# get_seismic_bumps()[:4]
# get_car_evaluation()[:4]
# get_zoo()[:4]
y_cl = one_hot(y)

X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y_cl, tp=0.6, vp=0.2)


fca = FCA(X_train)
l = fca.calculate_disjunctive_lattice(num_level=5)
# l = fca.calculate_lattice(num_level=4)
fca.save_lattice()
l = fca.load_lattice()

diag = Diagram(lattice=l, num_attributes=X_train.shape[1], X=X)
diag.calculate_child_concepts()
diag.save_child_concepts()
diag.load_child_concepts()



num_level = 3
concept_indices = diag.select_pure(y_cl, num_level)


for i in concept_indices:
    print i, " ", diag.child_concepts[i]

import keras
from copy import copy

input1 = keras.layers.Input(shape=(X_train.shape[1],))

array_of_layers = []
used_neurons = set()


class neuron:
    def __init__(self, layer, number):
        self.layer = layer
        self.number = number
        

for i in range(0, len(concept_indices)):
    if (len(diag.child_concepts[concept_indices[i]]) == 0):
        array_of_layers.append(   neuron(keras.layers.Dense(1, activation='relu')(input1),
                                concept_indices[i]))
        
        used_neurons.add(concept_indices[i])

#we have to repeat this n times where n = ???
for i in range(0, len(concept_indices)):
    if (set(diag.child_concepts[concept_indices[i]]) <= used_neurons and
        not concept_indices[i] in used_neurons):
        
        
        print("added: ", concept_indices[i], " ", used_neurons)
        current = []
        for j in diag.child_concepts[concept_indices[i]]:
            for k in range(0, len(array_of_layers)):
                if(array_of_layers[k].number == j):
                    current.append(array_of_layers[k].layer)
        
        array_of_layers.append( neuron(keras.layers.add(copy(current)),
                                       -1))
        
        array_of_layers.append(neuron(keras.layers.Dense(1, activation = 'relu')( 
                        array_of_layers[-1].layer), concept_indices[i]))
        used_neurons.add(concept_indices[i])
                

have_children = [0 for i in range(0, len(diag.child_concepts.keys()))]
for i in range(0, len(diag.child_concepts.keys()) ):
    if(i in  set(concept_indices) ):
        for j in range(0, len(diag.child_concepts[i])):
            have_children[diag.child_concepts[i][j]]  = 1
            
        

not_have_children = []   
for i in range(0, len( have_children)):
    if(have_children[i] == 0 and i in set(concept_indices)):
        not_have_children.append(i)
    

last_layer = []

for i in range(0, len(not_have_children)):
    for j in range(0, len(array_of_layers)):
       
        if(array_of_layers[j].number == not_have_children[i]):
            
            
            last_layer.append(array_of_layers[j].layer)

last_ = keras.layers.concatenate(last_layer)
outp = keras.layers.Dense(2, )(last_)
model = Model(inputs=input1, outputs=outp)


for i in range(0, len(array_of_layers)):
    print(array_of_layers[i].layer)
    
    
from keras.utils import plot_model
plot_model(model, to_file='model.png')

model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(X_train, y_train)
z = model.predict(X_test)
for i in range(len(z)):
    print(z[i], " ", y_test[i])



for i in range(0, len(concept_indices)):
    print (concept_indices[i], diag.child_concepts[concept_indices[i]])
    
current = []
current.append(array_of_layers[0].layer)
current.append(array_of_layers[1].layer)


t = keras.layers.concatenate(
        
a  = set()
a.add(2)
a.add(3)

b = set()
b.add(2)

len(X_train[0])






num_level = 3











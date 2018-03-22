#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:59:32 2018

@author: dmitriy
"""

import numpy as np
import matplotlib.pyplot as plt

import math
from sklearn.datasets import load_digits




from copy import copy

class interval:
    def __init__(self, a, b):
        self.a = a
        self.b = b


    
def check_canonicity(a):
    if(a == sorted(a)):
        return True
    else:
        return False        

class pattern_structure:
    def __init__(self, data):
        self.data = data
        
    
    def check_system_is_closed(self, array_of_max_min, system):
        to_close = []
        if(len(self.data[0]) != len(array_of_max_min)):
            #print("Error!")
            return 0
        else:
            for i in range(0, len(self.data)):
                if(i not in set(system)):
                    for j in range(0, len(self.data[0])):
                        if(self.data[i][j] > array_of_max_min[j][0] and 
                           self.data[i][j] < array_of_max_min[j][1] and not i in set(to_close)):
                            
                            to_close.append(i)
                            
                        

        
        return to_close
    

    
    def get_full_closure(self, array_of_max_min, system):
        system2 = copy(system)
        array_of_max_min_ = copy(array_of_max_min)
        to_close = self.check_system_is_closed(array_of_max_min, system2)
        
        
        while(len(to_close) > 0):
            for i in range(0, len(to_close)):
                if(not to_close[i] in set(system)):
                    system2.append(to_close[i])
                
                for j in range(0, len(self.data[0])):
                    if(self.data[to_close[i]][j] < array_of_max_min[j][0]):
                        array_of_max_min_[j][0] = self.data[to_close[i]][j]

                    if(self.data[to_close[i]][j] > array_of_max_min[j][1]):
                        array_of_max_min_[j][1] = self.data[to_close[i]][j]
            
            to_close = self.check_system_is_closed(array_of_max_min_, system2)
            
        return system2, array_of_max_min_
    
    
    
    def build(self,  system, array_of_max_min):
        for i in range(system[-1] + 1, len(self.data)):
            system2 = copy(system)
            #print("parent:", system2)
            system2.append(i)
            
            for j in range(0, len(self.data[0])):
            
                array_of_max_min[j][0] = min(self.data[system2[-1]][j], array_of_max_min[j][0])
                array_of_max_min[j][1] = max(self.data[system2[-1]][j], array_of_max_min[j][1])
            #print("added i: ",i, " ")
            
            system2, array_of_max_min_ = self.get_full_closure(array_of_max_min, system2)
            #print(system2)
            if(check_canonicity(system2)):
                print("ancestor", system2)
                self.build(system2, array_of_max_min_)
                
    
    

        
def test_pattern_structure():
            
        
    data = [[1,2], 
            [3, 4],
            [5, 0],
            [-2,1]]
                   
    system = [0]
    array_of_max_min = [[1, 1], [2, 2]]         
    
    
    ps = pattern_structure(data)
    
    
    ps.build(system, array_of_max_min)
    
    
    c = [4, 8, 7, 1, 9]
    sorted(c)
    
    system2 = ps.get_full_closure(array_of_max_min, system)
                            
from sklearn.datasets import fetch_mldata



def get_max_min_start_with(data, index):
    result = []
    for i in range(0, len(data[index])):
        result.append([])
        result[i].append(data[index][i])
        result[i].append(data[index][i])
    return result


'''
def get_max_min(data, indexes):
    result = []
    for i in range(0, len(data[0])):
        result.append([])
        result[i].append(1000000)
        result[i].append(-1000000)
'''

        
def test_mnist_64():
    data = load_digits().data
    target = load_digits().target
    
    data2 = data[:30]
    data = data[:30]
    
    tmp = []
    for i in range(len(data2)):
        tmp.append(data2[i][10:11])
    for current_index in range(10):
        array_of_max_min = get_max_min_start_with(tmp, current_index)
        system = [current_index]
        
    
        ps = pattern_structure(tmp)
        
        ps.build(system, array_of_max_min)
    


        
    

        
        
        
                
                
    
            
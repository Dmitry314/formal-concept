#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:59:32 2018

@author: dmitriy
"""



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
            print("Error!")
            return 0
        else:
            for i in range(0, len(self.data)):
                if(i not in set(system)):
                    for j in range(0, len(self.data[0])):
                        if(self.data[i][j] > array_of_max_min[j][0] and 
                           self.data[i][j] < array_of_max_min[j][1]):
                            to_close.append(i)
                        

        
        return to_close
    

    
    def get_full_closure(self, array_of_max_min, system):
        system2 = copy(system)
        to_close = self.check_system_is_closed(array_of_max_min, system)
        while(len(to_close) > 0):
            for i in range(0, len(to_close)):
                system2.append(to_close[i])
                
                for j in range(0, len(self.data[0])):
                    if(data[to_close[i]][j] < array_of_max_min[j][0]):
                        array_of_max_min[j][0] = data[to_close[i]][j]

                    if(data[to_close[i]][j] > array_of_max_min[j][1]):
                        array_of_max_min[j][1] = data[to_close[i]][j]
            
            to_close = self.check_system_is_closed(array_of_max_min, system2)
               
        return system2, array_of_max_min
    
    
    
    def build(self,  system, aray_of_max_min):
        for i in range(system[-1] + 1, len(data)):
            system2 = copy(system)
            system2.append(i)
            system2, array_of_max_min_ = self.get_full_closure(system2, array_of_max_min)
            if(check_canonicity(system)):
                print(system2)
                self.build(system, array_of_max_min_)
                
    
    

        
        
        
        
    
data = [[1,2], 
        [3, 4],
        [5, 0],
        [-2,1]]
               
system = [0, 1]
array_of_max_min = [[1, 3], [3,4]]         


ps = pattern_structure(data)


ps.build(system, array_of_max_min)


c = [4, 8, 7, 1, 9]
sorted(c)

system2 = ps.get_full_closure(array_of_max_min, system)
                        
            
       
            


        
    

        
        
        
                
                
    
            
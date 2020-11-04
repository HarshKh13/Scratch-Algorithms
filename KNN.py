import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

datadir = "D:\Cats and Dogs images\kagglecatsanddogs_3367a\PetImages"
categories = ["cane","cavallo","elefante","farfalla","gallina","gatto",
              "mucca","pecora","ragno","scoiattolo"]
complete_data = []
img_size = 16

def create_data(datadir,categories):
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr,(img_size,img_size))
                complete_data.append([new_arr,class_num])
                
            except Exception as e:
                pass
            
create_data(datadir,categories)
tot_data = len(complete_data)

random.shuffle(complete_data)
x = []
y = []

for features,lables in complete_data:
    x.append(features)
    y.append(lables)
    
x = np.reshape(x,(-1,1,img_size,img_size))
y = np.reshape(y,-1)
mask = np.random.choice(tot_data,15000)
x = x[mask]
y = y[mask]
tot_data = 15000

data = {}

def split_data(x,y,num_train=(4*tot_data)//5,val_size=tot_data//5):
    mask = list(range(num_train))
    x_train = x[mask]
    y_train = y[mask]
    
    mask = list(range(num_train,tot_data))
    x_val = x[mask]
    y_val = y[mask]
    
    return x_train,y_train,x_val,y_val

x_train,y_train,x_val,y_val = split_data(x,y)
data['x_train'], data['y_train'] = x_train, y_train
data['x_val'], data['y_val'] = x_val, y_val

class KNN(object):
    
    def __init__(self):
        pass
    
    def train(self,x,y):
        
        self.x_train = x.reshape(x.shape[0],-1)
        self.y_train = y
        
    def com_dis(self,x):
        
        num_train = len(self.x_train)
        num_test = x.shape[0]
        
        dists = np.zeros((num_test,num_train))
        
        
        
        
        
        
        



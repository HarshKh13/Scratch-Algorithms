#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import random

#defining name of required directories
clear_sky_dir = 'Clear sky'
cloudy_sky_dir = 'Cloudy'
data1 = 'Data1'
data2 = 'Data2'

# function for retriving images as array when directory is given as argument of function
def get_data(datadir):
    complete_images = []
    for mini_dir in os.listdir(datadir):
        image_dir = os.path.join(datadir,mini_dir)
        for image_path in os.listdir(image_dir):
            image_path = os.path.join(image_dir,image_path)
            image = cv2.imread(image_path)
            complete_images.append(image)
    
    return complete_images

# "clear_sky_images_train" and "cloudy sky_dir_train" contains images of clear and cloudy 
# nights respectively

clear_sky_dir_train = os.path.join(data1,clear_sky_dir)
cloudy_sky_dir_train = os.path.join(data1,cloudy_sky_dir)
clear_sky_images_train = get_data(clear_sky_dir_train)
cloudy_sky_images_train = get_data(cloudy_sky_dir_train)
clear_sky_images_train = np.array(clear_sky_images_train)
cloudy_sky_images_train = np.array(cloudy_sky_images_train)

# The script below generates augmented images. Augmented images are the copy of original 
# images which have changed orientation, zoom, rotation etc.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range = 90,
                             horizontal_flip = True, vertical_flip = True,
                             shear_range = 0.2, zoom_range = 0.2, 
                             fill_mode = 'constant')

clear_sky_aug = None
cloudy_sky_aug = None
for path in os.listdir(clear_sky_dir_train):
    if path.startswith("augmented"):
        clear_sky_aug_train = os.path.join(clear_sky_dir_train,path)
        cloudy_sky_aug_train = os.path.join(cloudy_sky_dir_train,path)
        break
 
itr = 0    
for batch in datagen.flow(clear_sky_images_train,batch_size = 10,
                          save_to_dir = clear_sky_aug_train):
    
    itr += 1
    if itr > 100:
        break
    
itr = 0    
for batch in datagen.flow(cloudy_sky_images_train,batch_size = 10,
                          save_to_dir = cloudy_sky_aug_train):
    
    itr += 1
    if itr > 100:
        break   
    
# The script below reads all the images i.e original images and augmented images.   
clear_sky_images_train = get_data(clear_sky_dir_train)
cloudy_sky_images_train = get_data(cloudy_sky_dir_train)
clear_sky_images_train = np.array(clear_sky_images_train)
cloudy_sky_images_train = np.array(cloudy_sky_images_train)
clear_sky_labels_train = [0]*clear_sky_images_train.shape[0]
cloudy_sky_labels_train = [1]*cloudy_sky_images_train.shape[0]

# The "total_images_train" list contains all the images of cloudy and clear nights together 
# and "total_labels_train" is list of labels for the corresponding image in "total_images_train".

total_images_train = np.concatenate((clear_sky_images_train,cloudy_sky_images_train),axis=0)
total_labels_train = np.concatenate((clear_sky_labels_train,cloudy_sky_labels_train),axis=0)

#The script below randomly shuffles the list.
data = list(zip(total_images_train,total_labels_train))
random.shuffle(data)
total_images_train,total_labels_train = zip(*data)
total_images_train = np.array(total_images_train)
total_labels_train = np.array(total_labels_train)

# The script below crops the images so that only the center part is used for training the model".
total_cropped_images_train = []
for image in total_images_train:
    height,width = image.shape[:2]
    start_row, start_col = int(height* .25), int(width* .25)
    end_row, end_col = int(height*.75), int(width* .75)
    cropped_image = image[start_row:end_row, start_col:end_col]
    cropped_image = cv2.resize(cropped_image,(128,128))
    total_cropped_images_train.append(cropped_image)

total_cropped_images_train = np.array(total_cropped_images_train)

# Importing required libraries for construction of model.
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers

# VGG16 is already trained model. "model1" returns features representations as
# a vector of length 8192 for each images. The final shape of features is (N,8192) 
# where N is the total number of images used for training the model.

model = VGG16(include_top = False, input_shape = (128,128,3))  
output1 = Flatten()(model.layers[-1].output)
model1 = Model(inputs = model.inputs, outputs = output1)
model_svm = Model(inputs = model.inputs, outputs = model.layers[-2].output)

preprocessed_input = preprocess_input(total_cropped_images_train)
ann_features_train = model1.predict(preprocessed_input)

##########################################################################################
#svm_features_train = model_svm.predict(preprocessed_input)
#svm_features_train = tf.keras.layers.GlobalMaxPool2D()(svm_features_train)
#svm_features_train = np.array(svm_features_train)
###########################################################################################

# The features representations returnde by the model1 is fed to model 2 which retures two 
# numbers which are the probabilities of an image being classified as cloudy and clear image.
input2 = tf.keras.layers.Input(
        shape = (8192),name = 'model2_inputs', dtype = np.float32)
hidden1 = Dense(4096,activation='relu')(input2)
hidden1 = layers.Dropout(0.2)(input2)
hidden2 = Dense(1024,activation='relu')(hidden1)
hidden2 = layers.Dropout(0.2)(hidden2)
output2 = Dense(2,activation='softmax')(hidden2)
model2 = Model(inputs = input2, outputs = output2)  

# The script below trains the model2 for classification thereby increasing the overall accuracy.
x_train_ann = ann_features_train
y_train_ann = total_labels_train
x_train_svm = svm_features_train
y_train_svm = total_labels_train

opt = tf.optimizers.Adam()
model2.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = opt, metrics=["accuracy"])
    
num_epochs = 10
batch_size = 20
model2.fit(x_train_ann,y_train_ann,batch_size = batch_size,
           epochs = num_epochs)

##########################################################################################
#from sklearn import svm
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score

#poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train_svm, y_train_svm)
#rbf = svm.SVC(kernel='rbf', gamma=1, C=1).fit(x_train_svm, y_train_svm)
#linear = svm.SVC(kernel='linear', C=1).fit(x_train_svm, y_train_svm)
###########################################################################################

###########################################################################################

clear_sky_dir_test = os.path.join(data2,clear_sky_dir)
cloudy_sky_dir_test = os.path.join(data2,cloudy_sky_dir)
clear_sky_images_test = get_data(clear_sky_dir_test)
cloudy_sky_images_test = get_data(cloudy_sky_dir_test)
clear_sky_images_test = np.array(clear_sky_images_test)
cloudy_sky_images_test = np.array(cloudy_sky_images_test)


clear_sky_aug_test = None
cloudy_sky_aug_test = None

for path in os.listdir(clear_sky_dir_test):
    if path.startswith("augmented"):
        clear_sky_aug_test = os.path.join(clear_sky_dir_test,path)
        cloudy_sky_aug_test = os.path.join(cloudy_sky_dir_test,path)
        break
 
itr = 0    
for batch in datagen.flow(clear_sky_images_test,batch_size = 10,
                          save_to_dir = clear_sky_aug_test):
    
    itr += 1
    if itr > 40:
        break
    
itr = 0    
for batch in datagen.flow(cloudy_sky_images_test,batch_size = 10,
                          save_to_dir = cloudy_sky_aug_test):
    
    itr += 1
    if itr > 40:
        break
            
clear_sky_images_test = get_data(clear_sky_dir_test)
cloudy_sky_images_test = get_data(cloudy_sky_dir_test)
clear_sky_images_test = np.array(clear_sky_images_test)
cloudy_sky_images_test = np.array(cloudy_sky_images_test)
clear_sky_labels_test = [0]*clear_sky_images_test.shape[0]
cloudy_sky_labels_test = [1]*cloudy_sky_images_test.shape[0]

total_images_test = np.concatenate((clear_sky_images_test,cloudy_sky_images_test),axis=0)
total_labels_test = np.concatenate((clear_sky_labels_test,cloudy_sky_labels_test),axis=0)

data = list(zip(total_images_test,total_labels_test))
random.shuffle(data)
total_images_test,total_labels_test = zip(*data)
total_images_test = np.array(total_images_test)
total_labels_test = np.array(total_labels_test)

total_cropped_images_test = []
for image in total_images_test:
    height,width = image.shape[:2]
    start_row, start_col = int(height* .25), int(width* .25)
    end_row, end_col = int(height*.75), int(width* .75)
    cropped_image = image[start_row:end_row, start_col:end_col]
    cropped_image = cv2.resize(cropped_image,(128,128))
    total_cropped_images_test.append(cropped_image)

total_cropped_images_test = np.array(total_cropped_images_test)

preprocessed_input = preprocess_input(total_cropped_images_test)
ann_features_test = model1.predict(preprocessed_input)
svm_features_test = model_svm.predict(preprocessed_input)
svm_features_test = tf.keras.layers.GlobalMaxPool2D()(svm_features_test)
svm_features_test = np.array(svm_features_test)

x_test_ann = ann_features_test
y_test_ann = total_labels_test
x_test_svm = svm_features_test
y_test_svm = total_labels_test

y_pred_ann = model2.predict(x_test_ann)
y_pred_ann = np.argmax(y_pred_ann,axis=1)

##################################################################################
#y_pred_polysvm = poly.predict(x_test_svm)
#y_pred_rbfsvm = rbf.predict(x_test_svm)
#y_pred_linearsvm = linear.predict(x_test_svm)
###################################################################################

def accuracy(y_pred,y_test):
    acc = 0
    for i in range(y_test.shape[0]):
        if(y_test[i]==y_pred[i]):
            acc += 1
            
    acc /= y_test.shape[0]
    return acc

acc_ann = accuracy(y_pred_ann,y_test_ann)

###############################################################################
#acc_lin = accuracy(y_pred_linearsvm,y_test_svm)
#acc_rbf = accuracy(y_pred_rbfsvm,y_test_svm)
#acc_poly = accuracy(y_pred_polysvm,y_test_svm)
##############################################################################









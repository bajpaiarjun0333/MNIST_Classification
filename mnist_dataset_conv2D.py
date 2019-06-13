#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 22:23:57 2019

@author: bajpaiarjun0333
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#code to classify the mnist dataset
(X_train,Y_train),(X_test,Y_test)=tf.keras.datasets.mnist.load_data()

#cross verifying if data is loaded or not
rand_index=999
print("The Image Label Is",Y_train[rand_index])
plt.imshow(X_train[rand_index])

#Verified that the data has been loaded from the server
print("Shape of X is",X_train.shape)
#reshaping the shape of x to latest keras documented way
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
#Cross checking the shapes opf X_train ,X_test
print("Shape of X train and X test are ",X_train.shape,X_test.shape)
#changing the type of the input as float
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
#normalizing the image input
X_train=X_train/255
X_test=X_test/255

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
#creating the model
model=Sequential()
model.add(Conv2D(28,kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=X_train,y=Y_train,epochs=10)
model.evaluate(X_test,Y_test)

test_image_index=333
plt.imshow(X_test[test_image_index].reshape(28,28),cmap='Greys')
pred=model.predict(X_test[test_image_index].reshape(1,28, 28,1))
print(pred.argmax())    






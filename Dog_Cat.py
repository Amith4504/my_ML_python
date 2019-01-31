# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:35:39 2019

@author: Amith R
Image classification using Convolutional Neural Networks
"""
#DATA Pre-Processing
#data set already divided into test and training set
# encoding is not required
# feature scaling is required
# building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialising the CNN
# object of sequential class
classifier = Sequential()

# step1 - Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
# using different feature detectors

#step2- MAxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#size of feature map reduced by 2

#adding a second convolutional layer to 
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3 -Flattening
classifier.add(Flatten())

#step4 - FullConnection
 #fully connected layer
classifier.add(Dense(output_dim=128,activation = 'relu'))
classifier.add(Dense(output_dim=1,activation = 'sigmoid'))

# compliling the CNN
classifier.compile (optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])
 
# Fitting the CNN to dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#object used for augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000, #no of images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
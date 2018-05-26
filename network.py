
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Activation,Reshape, Flatten, Conv2D, Dropout, BatchNormalization, Input, MaxPooling2D, Conv2DTranspose
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os, os.path
from scipy.misc import imread
from scipy.misc import imshow
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


img_width, img_height = 150, 150


# In[3]:


train_data_dir = '/Users/tomkennedy/221project/training'
validation_data_dir = '/Users/tomkennedy/221project/validation'


# In[4]:


nb_train_samples = 599
nb_validation_samples = 75
epochs = 50
batch_size = 16


# In[5]:


input_tensor = Input(shape=(150,150,3))
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
print('Model loaded.')


# In[6]:


top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu',kernel_initializer='glorot_normal', bias_initializer='zeros'))
top_model.add(Dropout(0.4))
top_model.add(Dense(5, activation='softmax',kernel_initializer='zeros', bias_initializer='zeros'))


# In[7]:


model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


# In[8]:


for layer in model.layers[:25]:
    layer.trainable = False


# In[9]:


adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999,  decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['categorical_accuracy'])


# In[10]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[11]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[12]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[13]:


model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)


# In[14]:


model.save_weights("intial_model.h5")


# In[ ]:





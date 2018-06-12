
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import regularizers
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
from keras.utils import multi_gpu_model
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# In[2]:


img_width, img_height = 224, 224


# In[3]:


train_data_dir = '/home/tomkennedy/training'
validation_data_dir = '/home/tomkennedy/validation'


# In[4]:


nb_train_samples = 624
nb_validation_samples = 78
epochs = 20
batch_size = 16


# In[5]:


image_net = keras.applications.VGG16(weights='imagenet', include_top=True, input_shape = None)
print('Model loaded.')
base_model = image_net
base_model.layers.pop()
base_model.layers.pop()
base_model.layers.pop()

num_layers_not_train = len(base_model.layers) - 9

new_model = Sequential()
for layer in base_model.layers:
    new_model.add(layer)
    
base_model.summary()
for layer in base_model.layers[:num_layers_not_train9]:
    layer.trainable = False
    
base_model.summary()
output = base_model.layers[len(base_model.layers)-1].output_shape
output


# In[6]:


top_model = Sequential()
top_model.add(Dense(256,input_shape = output, activation='relu',kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(1)))
top_model.add(Dropout(0.4))
top_model.add(Dense(5, activation='softmax',kernel_initializer='zeros', bias_initializer='zeros'))


# In[7]:


new_model.add(top_model)
new_model.summary()



# In[8]:


adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999,epsilon = .1, decay=0.0)
new_model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['categorical_accuracy'])



# In[9]:



train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range =.2)

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)



# In[10]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)





# In[11]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[12]:


mod = new_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

new_model.save_weights("final_first_train.h5")



# In[13]:


cont_model = new_model


# In[14]:


cont_model.load_weights("final_first_train.h5")
cont_model.summary()


# In[15]:


adam_2 = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999,epsilon = .1, decay=0.0)
cont_model.compile(loss='categorical_crossentropy',
              optimizer=adam_2,
              metrics=['categorical_accuracy'])


# In[16]:


mod_2 = cont_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=10,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

cont_model.save_weights("final_model.h5")


# In[17]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import seaborn as sn

testing_datagen = ImageDataGenerator(rescale=1. / 255)
testing_generator = testing_datagen.flow_from_directory(
    "/home/tomkennedy/test",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = False,
    class_mode='categorical')


new_model.load_weights("final_model.h5")
probabilities = new_model.predict_generator(testing_generator)
y_pred = np.array([(np.argmax(l)) for l in probabilities])
y_pred


# In[18]:


y_true = []
for name in testing_generator.filenames:
    if ("Leonardo_da_Vinci" in name): y_true.append(0)
    if ("Parmigianino" in name): y_true.append(1)
    if ("Sofonisba_Anguissola" in name): y_true.append(2)
    if ("Tintoretto" in name): y_true.append(3)
    if ("Titian" in name): y_true.append(4)
y_true = np.array(y_true)


# In[19]:


q = confusion_matrix(y_true, y_pred)


# In[20]:


q


# In[21]:


qnormalized = q.astype('float') / q.sum(axis=1)[:, np.newaxis]
qnormalized


# In[22]:


data_frame = pd.DataFrame(qnormalized, index = [i for i in validation_generator.class_indices.keys()],
                  columns = [i for i in validation_generator.class_indices.keys()])
sn.heatmap(data_frame, annot=True)


# In[23]:


x = classification_report(y_true,y_pred)
x


#This is a sample solution to the behavior cloning project 
#from the self-driving car nano degree of Udacity


import numpy as np
import matplotlib.pyplot as plt
import csv
import os


#First, import the logfile
#Replace this with your own logfile
training_file='.\datax\data\driving_log.csv'

training_data=csv.reader(open(training_file))

center=[]
left=[]
right=[]
steering=[]
throttle=[]
brake=[]
speed=[]
cont=0;
for data in training_data:
    if cont==0: 
        cont=cont+1;
        continue;
    else:
        cont=cont+1;
    center.append(data[0])
    left.append(data[1])
    right.append(data[2])
    steering.append(data[3])
    throttle.append(data[4])
    brake.append(data[5])
    speed.append(data[6])
print(len(center))



# Import tensorflow and keras

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda
from keras.layers import Cropping2D
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)


# Read the images


import matplotlib.image as mpimg
images=[];
steer=[];
for img_cont in range(len(center)):
    img_center_path='./datax/data/'+center[img_cont]
    img_left_path='./datax/data/'+left[img_cont]
    img_right_path='./datax/data/'+right[img_cont]

    image_center = mpimg.imread(img_center_path)
    image_left = mpimg.imread(img_left_path)
    image_right = mpimg.imread(img_right_path)
    
    images.append(image_center)



# In[47]:


print(len(images))
x_train=np.array(images)
y_train=np.array(steering)
print(y_train.shape)


# In[48]:


print(x_train.shape)


# Setup the model


model= Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(filters=6,strides=2, kernel_size=5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Convolution2D(filters=16,strides=3, kernel_size=3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.25))                                
model.add(Flatten())                     

model.add(Dense(units=512))
model.add(Activation('relu'))

model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=1))



# Train the model



model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_split=0.2,shuffle=True,epochs=50)


# Save the model


model.save('model.h5')


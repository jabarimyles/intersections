import pandas as pd

import json

from pandas.io.json import json_normalize



for n in range(1073):

    start=1000*n

    end=1000*(n+1)

    r = requests.get(url='https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_RoadCharacteristicsQtr/MapServer/0/query?where=OBJECTID>='+str(start)+'%20and%20OBJECTID<'+str(end)+'&outFields=*&outSR=4326&f=json').json()

    #print(r["features"][0]["attributes"])

    i=0

    df_tmp = pd.DataFrame()

    for f in r["features"]:

        df_tmp = df_tmp.append(json_normalize(r["features"][i]["attributes"]))

        i = i + 1



    if n==0:

        df_tmp.to_csv('/ndt/warehouse/ferries/saabur/ncdot_road_characteristics.csv', mode='a', index=None, header=True)

    else:

        df_tmp.to_csv('/ndt/warehouse/ferries/saabur/ncdot_road_characteristics.csv', mode='a', index=None, header=False)
 

 

32 Attachments
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:16:32 2019

@author: jamyle
"""

#plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import pandas as pd
#, fnmatch, cv2
# Imports
import os, image
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

intersections = pd.read_sas('C:/Users/jamyle/Documents/Traffic/traffic_intersection_stack.sas7bdat')
outdir = 'C:/Users/jamyle/Documents/Traffic/SatelliteImages'


data = intersections[['type', 'lat', 'long', 'angle', 'id']]
#data = data.sample(frac=1) #randomly shuffle dataframe

folderpath = "//sso.vsp.sas.com/amd/Customers/NC_Department_of_Transportation_NDT/Solution_Development/Traffic/TerrainOtsuImages/"

os.chdir("//sso.vsp.sas.com/amd/Customers/NC_Department_of_Transportation_NDT/Solution_Development/Traffic")

imageArray = [] #initialize empty arrays
imageLabel = []
    #imageArray = np.empty([128,128])
#for folderName in search:
#    imageList = []
#    imageList = os.listdir(topFolderPath + folderName)
ids = data['id']
i = 0 
#creates numerical input for CNN from otsu images
for inter_id in ids:
    img = Image.open(folderpath + "Terrain_Otsu_" + str(int(inter_id)) + ".png")
    img = img.resize((128, 128))
    img = np.asarray(img, dtype="float32")
    imageArray.append([img])
        #print(plt.imshow(img))
    imageLabel.append(data.ix[i,"type"])
    print("Image number " + str(i+1) + " out of 3414 just finished!")
    i = i+1
#imageFull = np.concatenate((np.array(imageArray), np.array(imageLabel)), axis=0)
   
            
train = np.array(np.asarray([i[0] for i in imageArray])).reshape(-1, 128, 128, 1) #creates "training" data to input, but it's really all the data

train_labels = np.array(imageLabel)
train_labels_dums = pd.get_dummies(train_labels)
train_labels_dums = np.array(train_labels_dums)

x_train, x_val, y_train, y_val = train_test_split(train, train_labels_dums, test_size=0.2) #splits into training and validation

########################################################################################################
# Training hyperparamters
EPOCHS = 20
BATCH_SIZE = 64

input_shape = (128, 128, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape)) 
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))

#model.add(Conv2D(256, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.025))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.025))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, 
              optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.0, amsgrad=False),
              metrics=['accuracy'])



######################## OPTION 1
#Talks to you while training. Gives in and out of sample accuracy. You should do some work here. Needs to output best validation instance, not final
model.fit(x_train, y_train, batch_size=BATCH_SIZE, 
              epochs=EPOCHS, verbose=1, 
              validation_data=(x_val, y_val)
              )

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    
test_score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))


######################## SAVE MODEL OPTION 1
from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("/insert/path/here/Model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/insert/path/here/Weights.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")



    

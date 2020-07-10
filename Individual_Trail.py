# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:01:28 2020

@author: BEL
"""
from pathlib import Path
import pathlib
import os , gc, glob, random

import pandas as pd
import numpy as np

import tensorflow as tf

import keras
import keras.backend as K
from  keras_applications import *
from keras_preprocessing import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization,Dropout, Lambda
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.models import Model
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

from sklearn.utils import shuffle
from skimage.morphology import label
from skimage import *
from skimage.data import *
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2

from IPython.display import display
from PIL import Image, ImageFile

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from subprocess import check_output
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.models import model_from_json


path, dirs, files = next(os.walk(r"C:\Users\BEL\Yaswanth\boat-types-recognition\boats"))

boat_types  = ['buoy', 'cruise ship', 'ferry boat', 'freight boat', 'gondola', 'infltable boat','kayak','paper boat', 'sailboat']
print("Boat Types",boat_types)

i = 0
j=0

X_data = []
Y_data = []

for boat in boat_types:
    files = glob.glob (r"C:/Users/BEL/Yaswanth/boat-types-recognition/boats/" + str(boat) + "/*.jpg")
    for myFile in files:
      img = Image.open(myFile)
      #img.thumbnail((width, height), Image.ANTIALIAS) # resizes image in-place keeps ratio
      img = img.resize((128,128), Image.ANTIALIAS) # resizes image without ratio
      img = np.array(img)
      if img.shape == ( 128, 128, 3):
        # Add the numpy image to matrix with all data
        X_data.append (img)
        Y_data.append (i)
    i += 1

print(X_data)
print(Y_data)

X = np.array(X_data)
Y = np.array(Y_data)

# Print shapes to see if they are correct
print("x-shape",X.shape,"y shape", Y.shape)

X = X.astype('float32') / 255.0
y_cat = to_categorical(Y_data, len(boat_types))

print("X shape",X,"y_cat shape", y_cat)
print("X shape",X.shape,"y_cat shape", y_cat.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)
print("The model has " + str(len(X_train)) + " inputs")

boats = []
number_of_boats = []

for dir in dirs:
  path2, dirs2, files2 = next(os.walk(r"C:/Users/BEL/Yaswanth/boat-types-recognition/boats/" + dir))
  boats.append(dir)
  number_of_boats.append(len(files2))

df = pd.DataFrame({'Boat Types':boats,
                   'N':number_of_boats})

df = df.sort_values(['N'], ascending=False)

df_actual = df.set_index('Boat Types')
# df_actual = df_actual.loc['boat_types']
df_actual = df_actual.sort_values(['N'], ascending=False)

fig, axes = plt.subplots(2,1, figsize=(14,14))  # 1 row, 2 columns
df_actual.plot(kind='bar',
               ax=axes[0],
               legend=False,
               color=[plt.cm.Paired(np.arange(len(df)))],
               width=0.85)
df_actual.plot('N',
               kind='pie',
               ax=axes[1],
               subplots=True)
plt.tight_layout()
plt.figure(figsize=(10, 10))

early_stop_loss = EarlyStopping(monitor='loss', patience=3, verbose=1)
early_stop_val_acc = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
model_callbacks=[early_stop_loss, early_stop_val_acc]

def load_CNN(output_size):
  K.clear_session()
  model = Sequential()
  model.add(Conv2D(128, (5, 5),input_shape=(128, 128, 3),activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  #model.add(BatchNormalization())

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  #model.add(BatchNormalization())

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  #model.add(BatchNormalization())

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(output_size, activation='softmax'))
  return model

model = load_CNN(9) #Number of Columns / Outputs
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0005),metrics=['accuracy'])
weights = model.get_weights()

batch_sizes = [4, 8, 16, 32, 64, 128]

histories_acc = []
histories_val_acc = []
histories_loss = []
histories_val_loss = []

for batch_size in batch_sizes:

  model.set_weights(weights)

  h=model.fit(X_train,y_train,
              batch_size=batch_size,
              epochs=3,
              verbose=0,
              callbacks=[early_stop_loss],
              shuffle=True,
              validation_data=(X_test, y_test))

  print(h.history.keys())

  histories_acc.append(h.history['accuracy'])
  histories_val_acc.append(h.history['val_accuracy'])
  histories_loss.append(h.history['loss'])
  histories_val_loss.append(h.history['val_loss'])

histories_acc = np.array(histories_acc)
histories_val_acc = np.array(histories_val_acc)
histories_loss = np.array(histories_loss)
histories_val_loss = np.array(histories_val_loss)
print('histories_acc',histories_acc,
      'histories_loss', histories_loss,
      'histories_val_acc', histories_val_acc,
      'histories_val_loss', histories_val_loss)

image_number = random.randint(0,len(X_train))
print(image_number)

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict([X_train[image_number].reshape(1, 128,128,3)])

def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(12,12))
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

display_activation(activations, 4, 4, 4)

plt.figure(figsize=(8, 8))
plt.imshow(X_train[image_number])

acc_bs = histories_acc
val_bs = histories_val_acc

plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for b in acc_bs:
  plt.plot(b)
  plt.title('Accuracy ')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['4', '8', '16', '32', '64', '128'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for z in val_bs:
  plt.plot(z)
  plt.title('Validation accuracy')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

# model.save('individual_model.h5')
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
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
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])



# boat_types_recognition =['boats', 'buoy', 'cruise ship', 'ferry boat', 'freight boat', 'gondola', 'inflatable boat', 'kayak', 'paper boat', 'sailboat']
# for boat_types in boat_types_recognition:
#     print(boat_types)
#     folder = glob.glob (r"C:/Users/BEL/Yaswanth/boat-types-recognition/" + str(boat_types))
#     print(folder)
#     for boat in folder:
#         print(boat)


# for boat_types in boat_types_recognition:
#     print(boat_types)
#     folder = glob.glob (r"C:/Users/BEL/Yaswanth/boat-types-recognition/" + str(boat_types))
#     for boat in boat_types:
#         files = glob.glob (str(boat) + "/*.jpg")
#         for myFile in files:
#             img = Image.open(myFile)
#             #img.thumbnail((width, height), Image.ANTIALIAS) # resizes image in-place keeps ratio
#             img = img.resize((128,128), Image.ANTIALIAS) # resizes image without ratio
#             img = np.array(img)
#             if img.shape == ( 128, 128, 3):
#                 # Add the numpy image to matrix with all data
#                 X_data.append (img)
#                 Y_data.append (j)
#         j += 1
#         Y_data.append(i)
#     i +=1


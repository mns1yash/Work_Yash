# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:41:57 2020

@author: BEL
"""
from pathlib import Path
import pathlib, os , gc, glob, random,keras, cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from keras_applications import *
from keras_preprocessing import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, layer, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization,Dropout, Lambda
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from skimage.morphology import label
from skimage import *
from skimage.data import *
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from IPython.display import display
from PIL import Image, ImageFile
from subprocess import check_output
from mpl_toolkits.axes_grid1 import ImageGrid
#___________________________________________________________________________________________________
# from  fastai import *
# print(os.listdir(r"C:\Users\BEL\Yaswanth\boat-types-recognition"))
# data_dir = r"C:\Users\BEL\Yaswanth\boat-types-recognition\boats"
# print(data_dir)
# files = pathlib.PureWindowsPath.__format__(${1:self}, format_spec)("C:/Users/BEL/Yaswanth/boat-types-recognition/boats/"+ str(boat) + "/*.jpg")

curr_wd = os.getcwd()
print("Current Woring Directory",curr_wd)
path, dirs, files = next(os.walk(r"C:\Users\BEL\Yaswanth\boat-types-recognition\boats"))
boat_types  = ['buoy', 'cruise ship', 'ferry boat', 'freight boat', 'gondola', 'infltable boat','kayak','paper boat', 'sailboat']
print("Boat Types",boat_types)

i = 0
X_data = []
Y_data = []
for boat in boat_types:
    files = glob.glob (r"C:/Users/BEL/Yaswanth/boat-types-recognition/boats/" + str(boat) + "/*.jpg")
    for myFile in files:
      img = Image.open(myFile)
      #img.thumbnail((width, height), Image.ANTIALIAS) # resizes image in-place keeps ratio
      img = img.resize((128,128), Image.ANTIALIAS) # resizes image without ratio
      img = np.array(img)
      if img.shape == (128, 128, 3):
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

# fig, axes = plt.subplots(2,2, figsize=(14,14))  # 1 row, 2 columns
fig, axes = plt.subplots(2,1, figsize=(14,14))  # 1 row, 2 columns

# df.plot('Boat Types',
#         ax=axes[0,0],
#         kind='bar',
#         legend=False,
#         color=[plt.cm.Paired(np.arange(len(df)))],
#         width=0.95)

# df_actual.plot(kind='bar',
#                ax=axes[0,1],
#                legend=False,
#                color=[plt.cm.Paired(np.arange(len(df)))],
#                width=0.95)
df_actual.plot(kind='bar',
               ax=axes[0],
               legend=False,
               color=[plt.cm.Paired(np.arange(len(df)))],
               width=0.85)

# df.plot('Boat Types',
#         'N',
#         kind='pie',
#         labels=df['Boat Types'],
#         ax=axes[1,0])

# df_actual.plot('N',
#                kind='pie',
#                ax=axes[1,1],
#                subplots=True)

df_actual.plot('N',
               kind='pie',
               ax=axes[1],
               subplots=True)
plt.tight_layout()

# plt.close('all')
plt.figure(figsize=(10, 10))

#___________________________________________________________________________________________________
#?

for i in range(25):
  # Plot the images in a 4x4 grid
  plt.subplot(5, 5, i+1)

  # Plot image [i]
  plt.imshow(X[i])

  # Turn off axis lines
  cur_axes = plt.gca()
  cur_axes.axes.get_xaxis().set_visible(False)
  cur_axes.axes.get_yaxis().set_visible(False)


#___________________________________________________________________________________________________
#Load CNN

def load_CNN(output_size):
  # K.clear_session()
  model = Sequential()
  model.add(Conv2D(128, (6, 6),input_shape=(128, 128, 3),activation='relu'))
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

#___________________________________________________________________________________________________
#Call Backs
early_stop_loss = EarlyStopping(monitor='loss', patience=3, verbose=1)
early_stop_val_acc = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
model_callbacks=[early_stop_loss, early_stop_val_acc]

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)
print("The model has " + str(len(X_train)) + " inputs")

#___________________________________________________________________________________________________
#SUMMARY AND COMPILE

model_1 = load_CNN(9) #Number of Columns / Outputs
model_1.summary()
model_1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0005),metrics=['accuracy'])
weights = model_1.get_weights()

#___________________________________________________________________________________________________
#BATCH PARAMETER

batch_sizes = [4, 8, 16, 32, 64, 128]
# batch_sizes = [4, 8]

histories_acc = []
histories_val_acc = []
histories_loss = []
histories_val_loss = []

for batch_size in batch_sizes:

  model_1.set_weights(weights)

  h=model_1.fit(X_train,y_train,
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

#___________________________________________________________________________________________________
#Learning Parameters

# learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
learning_rates = [0.01,0.001,0.0001]

lrsHistories_acc = []
lrsHistories_val = []
lrsHistories_loss = []
lrsHistories_val_loss = []

for lr in learning_rates:

    lr_model=load_CNN(9)

    lr_model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    lr_h = lr_model.fit(X_train, y_train,
                  batch_size=16,
                  epochs=3,
                  verbose=0,
                  callbacks=[early_stop_loss],
                  shuffle=True,
                  validation_data=(X_test, y_test))
    print(lr_h.history.keys())

    lrsHistories_acc.append(lr_h.history['accuracy'])
    lrsHistories_val.append(lr_h.history['val_accuracy'])
    lrsHistories_loss.append(lr_h.history['loss'])
    lrsHistories_val_loss.append(lr_h.history['val_loss'])

lrsHistories_acc = np.array(lrsHistories_acc)
lrsHistories_val_acc = np.array(lrsHistories_val)
lrsHistories_loss = np.array(lrsHistories_loss)
lrsHistories_val_loss = np.array(lrsHistories_val_loss)
print('lrshistories_acc', lrsHistories_acc,
      'lrshistories_loss', lrsHistories_loss,
      'lrshistories_val_acc', lrsHistories_val_acc,
      'lrshistories_val_loss', lrsHistories_val_loss)

#___________________________________________________________________________________________________
#Opimizers

optimizers = ['SGD(lr=0.0001)',
              'SGD(lr=0.0001, momentum=0.3)',
              'SGD(lr=0.0001, momentum=0.3, nesterov=True)',
              'Adam(lr=0.0001)',
              'Adagrad(lr=0.0001)',
              'RMSprop(lr=0.0001)']
optimizeList_acc = []
optimizeList_val = []
optimizeList_loss = []
optimizeList_val_loss = []

for opt_name in optimizers:

    opt_model=load_CNN(9)

    opt_model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt_name),
                  metrics=['accuracy'])
    opt_h = opt_model.fit(X_train, y_train,
                  batch_size=16,
                  epochs=3,
                  verbose=0,
                  callbacks=[early_stop_loss],
                  shuffle=True,
                  validation_data=(X_test, y_test))
    optimizeList_acc.append(opt_h.history['accuracy'])
    optimizeList_val.append(opt_h.history['val_accuracy'])
    optimizeList_loss.append(opt_h.history['loss'])
    optimizeList_val_loss.append(opt_h.history['val_loss'])

optimizeList_acc = np.array(optimizeList_acc)
optimizeList_val = np.array(optimizeList_val)
optimizeList_loss = np.array(optimizeList_loss)
optimizeList_val_loss = np.array(optimizeList_val_loss)
print("optimizeList_acc",optimizeList_acc,"optimizeList_val",optimizeList_val,"optimizeList_loss",
      optimizeList_loss,"optimizeList_val_loss",optimizeList_val_loss)
#___________________________________________________________________________________________________
#Display Ativation

image_number = random.randint(0,len(X_train))
print(image_number)

layer_outputs = [layer.output for layer in model_1.layers]
activation_model = Model(inputs=model_1.input, outputs=layer_outputs)
x_train_dim = np.expand_dims(X_train[image_number],axis =0)
activations = activation_model.predict(x_train_dim)

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

#___________________________________________________________________________________________________
#Plots

plt.figure(figsize=(8, 8))
plt.imshow(X_train[image_number])

display_activation(activations, 4, 4, 4)

acc_lr = lrsHistories_acc
val_lr = lrsHistories_val_acc

acc_bs = histories_acc
val_bs = histories_val_acc

acc_opt = optimizeList_acc
val_opt = optimizeList_val

#___________________________________________________________________________________________________

plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for b in acc_bs:
  plt.plot(b)
  plt.title('Accuracy for different batch sizes')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['4', '8', '16', '32', '64', '128'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

#___________________________________________________________________________________________________

plt.subplot(212)
for z in val_bs:
  plt.plot(z)
  plt.title('Validation accuracy for different batch sizes')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

#___________________________________________________________________________________________________

plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in acc_lr:
  plt.plot(x)
  plt.title('Accuracy for different learning rates')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

#___________________________________________________________________________________________________

plt.subplot(212)
for y in val_lr:
  plt.plot(y)
  plt.title('Validation accuracy for different learning rates')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

#___________________________________________________________________________________________________

plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in acc_opt:
  plt.plot(x)
  plt.title('Accuracy for different optimizers')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

#___________________________________________________________________________________________________

plt.subplot(212)
for y in val_opt:
  plt.plot(y)
  plt.title('Validation accuracy for different optimizers')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], bbox_to_anchor= (1.05,1), loc='lower right' )
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

#___________________________________________________________________________________________________
#DATA AUG

dataaug_model=load_CNN(9)
dataaug_model.compile(loss='binary_crossentropy',
              optimizer=Adagrad(0.0005),
              metrics=['accuracy'])

dataaug_h = dataaug_model.fit(X_train, y_train,
              batch_size=8,
              epochs=3,
              verbose=0,
              callbacks=[early_stop_loss],
              shuffle=True,
              validation_data=(X_test, y_test))

y_pred = dataaug_model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis = 1)
y_true = np.argmax(y_test,axis = 1)

#___________________________________________________________________________________________________

plt.plot(dataaug_h.history['accuracy'], label='accuracy')
plt.plot(dataaug_h.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['acc','val_acc'], loc='best')

#___________________________________________________________________________________________________

con_matrix = confusion_matrix(y_true, y_pred_classes, labels=[0,1,2,3])
plt.figure(figsize=(10,10))
plt.title('Prediction of boat types')
sns.heatmap(con_matrix, annot=True, fmt="d", linewidths=.5)

#___________________________________________________________________________________________________
#Image Data Generator For Data Augmentation

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = datagen.flow(X_train, y_train, batch_size=256)
validation_generator = datagen.flow(X_test, y_test, batch_size=256)
#___________________________________________________________________________________________________
plt.close('all')
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=16):
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X_batch[i])
        # Turn off axis lines
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
    break

#___________________________________________________________________________________________________
#Data Aug Batch

data_aug_model = load_CNN(9)
data_aug_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
weights_aug = data_aug_model.get_weights()
batch_sizes = [8, 16, 32, 64, 128, 256]
data_augmentation_bs_acc = []
data_augmentation_bs_val = []
for batch_size in batch_sizes:
  train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
  validation_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
  data_aug_model.set_weights(weights_aug)
  data_aug_h = data_aug_model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=3,
                    validation_data=validation_generator,
                    validation_steps=10,
                    callbacks=[early_stop_loss],
                    verbose=0)

  data_augmentation_bs_acc.append(data_aug_h.history['accuracy'])
  data_augmentation_bs_val.append(data_aug_h.history['val_accuracy'])
data_augmentation_bs_acc = np.array(data_augmentation_bs_acc)
data_augmentation_bs_val = np.array(data_augmentation_bs_val)
#___________________________________________________________________________________________________
# Data Aug Learning Rates

data_augmentation_lr_acc = []
data_augmentation_lr_val = []

train_generator = datagen.flow(X_train, y_train, batch_size=16)
validation_generator = datagen.flow(X_test, y_test, batch_size=16)

for lr in learning_rates:
    aug_lr_model=load_CNN(9)
    aug_lr_model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    aug_lr_h = aug_lr_model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=3,
                    validation_data=validation_generator,
                    callbacks=[early_stop_loss],
                    verbose=0,
                    validation_steps=10)
    data_augmentation_lr_acc.append(aug_lr_h.history['accuracy'])
    data_augmentation_lr_val.append(aug_lr_h.history['val_accuracy'])
data_augmentation_lr_acc = np.array(data_augmentation_lr_acc)
data_augmentation_lr_val = np.array(data_augmentation_lr_val)

#___________________________________________________________________________________________________
#Data Aug Optimizers

data_augmentation_opt_acc = []
data_augmentation_opt_val = []

for opt_name in optimizers:

    aug_opt_model=load_CNN(9)

    aug_opt_model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt_name),
                  metrics=['accuracy'])
    aug_opt_h = aug_opt_model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=3,
                    validation_data=validation_generator,
                    callbacks=[early_stop_loss],
                    verbose=0,
                    validation_steps=10)

    data_augmentation_opt_acc.append(aug_opt_h.history['accuracy'])
    data_augmentation_opt_val.append(aug_opt_h.history['val_accuracy'])
data_augmentation_opt_acc = np.array(data_augmentation_opt_acc)
data_augmentation_opt_val = np.array(data_augmentation_opt_val)

#___________________________________________________________________________________________________
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for b in data_augmentation_bs_acc:
  plt.plot(b)
  plt.title('Accuracy for different batch sizes')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

#___________________________________________________________________________________________________
plt.subplot(212)
for z in data_augmentation_bs_val:
  plt.plot(z)
  plt.title('Validation accuracy for different batch sizes')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

#___________________________________________________________________________________________________
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in data_augmentation_lr_acc:
  plt.plot(x)
  plt.title('Accuracy for different learning rates')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

#___________________________________________________________________________________________________
plt.subplot(212)
for y in data_augmentation_lr_val:
  plt.plot(y)
  plt.title('Validation accuracy for different learning rates')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

#___________________________________________________________________________________________________
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in data_augmentation_opt_acc:
  plt.plot(x)
  plt.title('Data augmentation accuracy for different optimizers')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom-0.01, 1.01)   # set the ylim to bottom, top

#___________________________________________________________________________________________________
plt.subplot(212)
for y in data_augmentation_opt_val:
  plt.plot(y)
  plt.title('Data augmentation validation accuracy for different optimizers')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top

#___________________________________________________________________________________________________
#Last Model
last_model=load_CNN(9)
last_model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.0002),
              metrics=['accuracy'])

last_h = last_model.fit_generator(train_generator,
                steps_per_epoch=len(X_train) / 32,
                epochs=5,
                validation_data=validation_generator,
                #callbacks=[early_stop_loss],
                verbose=0,
                validation_steps=10)

#___________________________________________________________________________________________________
y_pred = last_model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis = 1)
y_true = np.argmax(y_test,axis = 1)

plt.plot(last_h.history['accuracy'], label='accuracy')
plt.plot(last_h.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['acc','val_acc'], loc='best')

con_matrix = confusion_matrix(y_true, y_pred_classes, labels=[0,1,2,3])

plt.figure(figsize=(10,10))
plt.title('Prediction of boat types')
sns.heatmap(con_matrix, annot=True, fmt="d", linewidths=.5)
#___________________________________________________________________________________________________

model_1.save('model_cnn.h5')
lr_model.save('lr_model.h5')
opt_model.save('opt_model.h5')
dataaug_model.save('dataaug_model.h5')
data_aug_model.save('data_aug_model.h5')
aug_lr_model.save('aug_lr_model.h5')
aug_opt_model.save('aug_opt_model.h5')
last_model.save('last_model.h5')

#___________________________________________________________________________________________________
import model_cnn
model_cnn.h5.to_json("model_1.json")

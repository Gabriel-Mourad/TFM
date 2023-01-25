# -*- coding: utf-8 -*-

import tensorflow as tf
import os,glob
import cv2
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Convolution2D, Dropout, Dense,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.applications import vgg16

# Directorios de trabajo
ROOT_DIR = '/gdrive/MyDrive/TFM'
os.chdir(ROOT_DIR)
DATA_DIR = "/gdrive/MyDrive/TFM/DataSet/vgg_data/Dataset/D_3b"
DATASET_RIGHT = os.path.join(DATA_DIR, 'RIGHT')
DATASET_LEFT = os.path.join(DATA_DIR, 'LEFT')
DATASET_CENTER = os.path.join(DATA_DIR, 'CENTER')
DATASET_NO_CENTER = os.path.join(DATA_DIR, 'NO_CENTER')
DATASET_DESTINO = os.path.join(DATA_DIR, "DESTINO")
DATASET_n_DESTINO = os.path.join(DATA_DIR, "NO_DESTINO")
DATASET_VALIDATION = os.path.join(DATA_DIR, "VALIDATION")

X = [] #GUARDAMOS IMAGENES
y = [] #GUARDAMOS LABEL


os.chdir(DATASET_RIGHT)
for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(224,224))
      X.append(img)
      y.append("RIGHT")

os.chdir(DATASET_LEFT)
for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(224,224))
      X.append(img)
      y.append("LEFT")

os.chdir(DATASET_CENTER)
for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(224,224))
      X.append(img)
      y.append("CENTER")

os.chdir(DATASET_DESTINO)
for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(224,224))
      X.append(img)
      y.append("DESTINO")
'''
os.chdir(DATASET_n_DESTINO)
for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(224,224))
      X.append(img)
      y.append("NO_DESTINO")
'''

print(y)

# Separación del conjunto de datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)
print ("Shape of an image in X_train: ", X_train[0].shape)
print ("Shape of an image in X_test: ", X_test[0].shape)
print(y_train)

# conversión del label (string) a clase numérica

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test) 

print("X_train Shape: ", X_train.shape) 
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape) 
print("y_test Shape: ", y_test.shape)

# Dimension de las imágenes de entrada
img_rows, img_cols = 224, 224 
# Modelo vgg-16 pre-configurado con los parámetros de ImageNet 
# en las capas inferiores
vgg = vgg16.VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Congelar las capas de la red que ya estan pre-entrenadas
for layer in vgg.layers:
    layer.trainable = False

# Imprimir capas
for (i,layer) in enumerate(vgg.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

# Crear la red neuronal clásica situada en la parte más alta del modelo
def lw(bottom_model, num_classes):
  top_model = bottom_model.output
  top_model = GlobalAveragePooling2D()(top_model)
  top_model = Dense(4096,activation='relu')(top_model)
  top_model = Dense(4096,activation='relu')(top_model)
  top_model = Dense(1000,activation='relu')(top_model)
  top_model = Dense(num_classes,activation='softmax')(top_model)
  return top_model

# Crear el modelo final
num_classes = 3
FC_Head = lw(vgg, num_classes)
model = Model(inputs = vgg.input, outputs = FC_Head)
print(model.summary())

# Configuración de hiperparámetros de entrenamiento

model.compile(optimizer='adam', loss = 'CategoricalCrossentropy',metrics = ['accuracy'])
history = model.fit(X_train,y_train,
                    epochs=5, 
                    validation_data=(X_test,y_test),
                    verbose = 1,
                    initial_epoch=0)

# Guardar configuración de pesos entrenados
os.chdir(DATA_DIR)
save_model(model,"modelo_desarrollo_2b")

# Crear gráfico de pérdida
import matplotlib.pyplot as plt
# %matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, train_loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.grid()
plt.figure()

plt.show()

import tensorflow as tf
import os,glob
import time
import cv2
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

# Directorio de trabajo
ROOT_DIR = '/home/gabriel/Desktop/TFM'
DATA_DIR = "/home/gabriel/Desktop/TFM/Dataset/D_1"
DATA_INFERENCE = os.path.join(ROOT_DIR, "INFERENCE")

# Cargar configuración de la red neuronal
os.chdir(DATA_DIR)
modelo = load_model("modelo_desarrollo_1")


def inference_d1():
    
    os.chdir(DATA_INFERENCE)
    
    # Procesado de la imagen capturada 
    X_inf = []
    img = cv2.imread("foto_inf.jpg")
    img = cv2.resize(img,(224,224))
    X_inf.append(img)
    X_inf = np.array(X_inf)
    
    # Prediccón del modelo
    y_hat = modelo.predict(X_inf)
    y_class = np.argmax(y_hat)
    
    return(y_class)

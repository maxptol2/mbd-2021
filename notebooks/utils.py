#!/usr/bin/env python
# coding: utf-8

# In[3]:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Common packages
import numpy as np
import pandas as pd
import warnings

# ML
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, Adam
import tensorflow as tf
from keras.activations import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight
import numpy
import keras_metrics

import sklearn
from sklearn.utils import class_weight

from xgboost import XGBClassifier
import sys
import folium
import json
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta as rt
import pyspark
import pandas as pd
from plotly import graph_objects as go
import numpy as np
import seaborn as sns
plt.style.use('ggplot')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def plot_cm(labels, predictions, p=0.5):
     
    """
    Se muestra la Matriz de confusión con casos reales vq predicción.
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    

def eval_model(training, model, test_X, test_y, field_name):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(3, figsize=(15,15))
    ax[0].plot(training.history['loss'], label="Loss")
    ax[0].plot(training.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Recall
    ax[1].plot(training.history['recall'], label="Recall")
    ax[1].plot(training.history['val_recall'], label="Validation Recall")
    ax[1].set_title('%s: Recall' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Recall')
    ax[1].legend()
    plt.tight_layout()
    
    # Precision
    ax[2].plot(training.history['precision'], label="Precision")
    ax[2].plot(training.history['val_precision'], label="Validation Precision")
    ax[2].set_title('%s: Precision' % field_name)
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Precision')
    ax[2].legend()
    plt.tight_layout()

    
def distancia_cliente_comercio(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Se calcula una nueva variable que consiste en la distancia que hay en kilometros entre el 
    lugar en donde vive el cliente y la localidad en donde se encuentra funcionando el comercio.
    """
    
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))
    
    
def read_data():
    """
    Se caran los datos
    """
    datatrain=pd.read_csv('fraudTrain.csv',header=0)
    datatrain = datatrain.drop(datatrain.columns[0], axis=1)
    datatest=pd.read_csv('fraudTest.csv',header=0)
    datatest = datatest.drop(datatest.columns[0], axis=1)
 
    return datatrain, datatest

def value_counts(data, attribute) :
    data[attribute].value_counts().plot(kind = 'bar')
    plt.ylabel('Count')
    plt.title(attribute)
    plt.show()
    
def split(datatrain):
    """ 
    Se realiza la división entre train y validación
    """
    datatrain_muestra, dataval_muestra = train_test_split(datatrain, test_size=0.2, random_state=24)
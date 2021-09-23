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
    datatrain=pd.read_csv('datasets/fraudTrain.csv',header=0)
    datatrain = datatrain.drop(datatrain.columns[0], axis=1)
    datatest=pd.read_csv('datasets/fraudTest.csv',header=0)
    datatest = datatest.drop(datatest.columns[0], axis=1)
 
    return datatrain, datatest

def Gra_bar(data, attribute) :
    plt.figure(figsize=(20,8))
    plt.tick_params(labelsize=23)
    data[attribute].value_counts().plot(kind = 'bar')
    plt.ylabel('Count')
    plt.title(attribute)
    plt.show()

def gender_binario(x):
    if x=='F':
        return 1
    if x=='M':
        return 0    
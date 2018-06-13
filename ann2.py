# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:15:24 2018

@author: keshu
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

#catagorical data encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])


labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[: , 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#part 2

import keras
from keras.models import Sequential
from keras.layers import Dense

#initalising ANN
classifier = Sequential()

#first input layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#output lauer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#compiling ANN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy' , metrics= ['accuracy'] )

#fitting ann
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred= (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
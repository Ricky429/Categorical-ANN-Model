#Categorical Model

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing data
dataset = pd.read_csv('Name of file.file format')
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9:13].values

#Splitting data set between test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

#Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Keras API
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

#Creating the ANN
classifier = Sequential()

#Creating the First Layer
classifier.add(Dense(output_dim = 4, init = 'he_uniform', activation = 'relu', input_dim = 9))
#Creating the Hidden Layers
classifier.add(Dense(output_dim = 13, init = 'he_uniform', activation = 'tanh', activity_regularizer = regularizers.l2(0.001)))
classifier.add(Dense(output_dim = 13, init = 'he_uniform', activation = 'tanh', activity_regularizer = regularizers.l2(0.001)))
classifier.add(Dense(output_dim = 13, init = 'he_uniform', activation = 'tanh', activity_regularizer = regularizers.l2(0.001)))
#Creating the Output Layer
classifier.add(Dense(output_dim = 4, init = 'he_uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'nadam', loss = 'categorical_crossentropy', metrics = ['binary_accuracy'])

#Training the model
classifier.fit(X_train, y_train, batch_size = 1, nb_epoch = 200)

#Predicting the test set
y_pred = classifier.predict(X_test)
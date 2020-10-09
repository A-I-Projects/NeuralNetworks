#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 06:52:18 2019

@author: aroonkp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 28, kernel_size = 3, input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
 
model.fit(X_train, y_train, epochs = 10)

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test[0].reshape(1, 28,28,1))
print(y_pred.argmax())
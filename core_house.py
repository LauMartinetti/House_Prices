import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from matplotlib import pyplot as plt
from tensorflow.data import Dataset
import numpy as np
import os
from tensorflow.keras.optimizers import Adam

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

test_np = test.to_numpy()
train_np = train.to_numpy()

targets = test_np[:,1]
inputs = test_np[:,2:]

train_y = train_np[:,1]
train_x = train_np[:,2:]

VAL_SPLIT = 0.5
EPOCHS = 20


### Model ###

model = Sequential()
model.add(Dense(160, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))


### Training ##

history=model.fit(inputs, targets, validation_split=VAL_SPLIT, epochs=EPOCHS)

plt.figure()
plt.grid()
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.plot(history.history['loss'], label = 'loss')
plt.yscale('log')

plt.legend()

plt.show()

print('_____________________________________')

test_loss, test_acc = model.evaluate(train_x, targets)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

print('_____________________________________')
print('_____________________________________')

### Test ##


predict= model.predict(train_x)


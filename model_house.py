import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from matplotlib import pyplot as plt
from tensorflow.data import Dataset
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
import seaborn as sns


#important values
VAL_SPLIT = 0.2
EPOCHS = 1000
LEARN_RATE=.0001
CORR_VALUE=.3





test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


for column in test.select_dtypes(include=['object']):
        train[column]=pd.Categorical(train[column])
        train[column]=train[column].cat.codes
        test[column]=pd.Categorical(test[column])
        test[column]=test[column].cat.codes

mask = abs(train.corr()['SalePrice'])> CORR_VALUE
col_mask = train.columns[mask]

train_correlated=train[col_mask]

#Data samples
inputs=train_correlated.drop('SalePrice', axis='columns')
targets = train["SalePrice"]
train_id = train["Id"]

test_id=test['Id']
test_inputs=test[inputs.columns]


#Correlation figures
sns.set(font_scale=0.75)
sns.heatmap(train.corr(), vmax=.8, square=True, xticklabels=True, yticklabels=True,linewidth=0.05,linecolor='black')

plt.figure()
sns.heatmap(inputs.corr(), vmax=.8, square=True, xticklabels=True, yticklabels=True,linewidth=0.05,linecolor='black')



### Model ###

model = Sequential()

model.add(Dense(10, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='MSE', optimizer=Adam(learning_rate=LEARN_RATE), metrics=['accuracy'])

### Training ##

history=model.fit(inputs, targets, validation_split=VAL_SPLIT, epochs=EPOCHS)

plt.figure()
plt.grid()
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.plot(history.history['loss'], label = 'loss')
plt.yscale('log')

plt.legend()


print('_____________________________________')

test_loss, test_acc = model.evaluate(inputs, targets)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

print('_____________________________________')
print('_____________________________________')

### Test ##

predict= model.predict(test_inputs)

plt.show()
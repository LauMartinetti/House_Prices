import os
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt


#important values
VAL_SPLIT = 0.3
EPOCHS = 50
LEARN_RATE=.001
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
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.legend(fontsize=13)
plt.savefig('losses.png', dpi=10000)


print('_____________________________________')

test_loss, test_acc = model.evaluate(inputs, targets)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

print('_____________________________________')
print('_____________________________________')


path = os.getcwd()
model.save(path)
### Test ##

predict= model.predict(test_inputs)

plt.show()

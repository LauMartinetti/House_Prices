import os
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels


#model values
VAL_SPLIT = 0.3
EPOCHS = 100
learn_rate=[.1, .1, .1, .1, .1, .1, .01, .01, .01]
corr_value=[0., 0., 0., 0., 0., 0., 0., .2, .4]
nodes= [[40,0,0], [40,20,0], [40,20,10], [200,100,50], [50, 100, 400], [40,20,10], [40,20,10], [40,20,10], [40,20,10]]
models_values = zip(learn_rate, corr_value, nodes)


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


for column in test.select_dtypes(include=['object']):
    train[column]=pd.Categorical(train[column])
    train[column]=train[column].cat.codes
    test[column]=pd.Categorical(test[column])
    test[column]=test[column].cat.codes

train_id = train["Id"]
test_id=test['Id']


i=0
### Model ###
for LEARN_RATE, CORR_VALUE, NODES in models_values:

    mask = abs(train.corr()['SalePrice'])> CORR_VALUE
    col_mask = train.columns[mask]

    train_correlated=train[col_mask]

    #Data samples
    inputs=train_correlated.drop('SalePrice', axis='columns')
    targets = train["SalePrice"]
    test_inputs=test[inputs.columns]


    model = Sequential()
    model.add(Dense(NODES[0], activation='relu'))
    if(NODES[1] != 0):
        model.add(Dense(NODES[1], activation='relu'))
        if (NODES[2] != 0):
            model.add(Dense(NODES[2], activation='relu'))
    model.add(Dense(1, activation='linear'))


    model.compile(loss='MSE', optimizer=Adam(learning_rate=LEARN_RATE), metrics=['accuracy'])

    history=model.fit(inputs, targets, validation_split=VAL_SPLIT, epochs=EPOCHS)


    plt.figure()
    plt.grid()
    text = "Loss \n Hidden layers' nodes= ({},{},{})\n Learning rate = {}\n Correlation value = {} ".format(NODES[0],NODES[1],NODES[2],LEARN_RATE,CORR_VALUE)

    plt.plot(history.history['loss'], label = text)
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right', fontsize = 8)

    name = 'model{}.png'.format(i)
    plt.savefig(name, dpi=1000)
    i=i+1
#plt.show()

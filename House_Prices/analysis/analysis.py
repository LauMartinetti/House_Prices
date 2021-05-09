import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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


#Correlation figures
sns.set(font_scale=0.75)
sns.heatmap(train.corr(), vmax=.8, square=True, xticklabels=True, yticklabels=True,linewidth=0.05,linecolor='black')

plt.figure()
sns.heatmap(inputs.corr(), vmax=.8, square=True, xticklabels=True, yticklabels=True,linewidth=0.05,linecolor='black')

plt.show()

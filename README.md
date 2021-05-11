# Model for prediction of houses prices
[TOC]

## Folders
### House_Prices
In this folder you will find:
- the folder "**analysis**"
- the folder "**variables**"
- **data_description.txt**: a text file in which there are specified and explained all datas
- **model_house.py**: the code implemented for the model
- **saved_model.pb**: the trained model
- **test.csv**: all the data used to test the model
- **train.csv** all the data used to train the model
#### analysis
In this folder you will find all codes used to evaluate and analyze the model and the related images.
-  **model_analysis**: analysis of the model changing the values
- **analysis.py**: code that generate the heatmap to visualize the correlation between the properties of the houses
- **correlations.png**: heatmap of the correlation between all the properties-
- **correlation_revised.png**: heatmap of the correlation between the properties with correlation >0.3 
- all the **model[].png** : representation of the loss of the training set and validation set for models with different values.

#### variables

## Libraries
The following additional libraries are used:
- matplotlib (in "model_analysis.py" and "model_house.py")
- tensorflow.keras
- pandas
- seaborn (in "analysis.py")



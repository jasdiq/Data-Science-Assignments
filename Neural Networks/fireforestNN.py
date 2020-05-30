# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:06:10 2020

@author: HP
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:19:45 2020

@author: HP
"""
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Reading data 
fire = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Neural Networks\\fireforests.csv")
fire.head()
fire.shape
objList = fire.select_dtypes(include = "object").columns
print (objList)
#labelencoder=LabelEncoder()
le = LabelEncoder()

for feat in objList:
    fire[feat] = le.fit_transform(fire[feat].astype(str))
    print (fire.info())
fire.columns
fire.drop(fire.columns[[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29]], axis = 1, inplace = True)
fire.head()
fire.columns

fire=fire.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,10]]
fire.columns

cont_model=Sequential()
cont_model.add(Dense(50, input_dim=29,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer="adam", metrics= ["mse"] )

column_names = list(fire.columns)
predictors = column_names[0:29]
target = column_names[29]


first_model=cont_model
first_model.fit(np.array(fire[predictors]),np.array(fire[target]),epochs=10)
pred_train = first_model.predict(np.array(fire[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-fire[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,fire[target],"bo")
np.corrcoef(pred_train,fire[target]) # we got high correlation 

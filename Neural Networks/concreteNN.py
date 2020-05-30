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



# Reading data 
Concrete = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Neural Networks\\concrete.csv")
Concrete.head()
Concrete.shape


cont_model=Sequential()
cont_model.add(Dense(50, input_dim=8,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer="adam", metrics= ["mse"] )

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]


first_model=cont_model
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=10)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target]) # we got high correlation 

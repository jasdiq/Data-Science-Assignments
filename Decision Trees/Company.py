# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:26:36 2020

@author: HP
"""


import pandas as pd
import matplotlib.pyplot as plt
comp = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Decision Trees\\Company_Data.csv")
labels = ["low", "medium", "high"]
bins=[0,6,12,17]
comp["Sales"]=pd.cut(comp['Sales'], bins=bins, labels=labels)
comp['ShelveLoc'].unique()
comp['Sales'].unique()
comp.Sales.value_counts()
comp.isnull().sum()
comp['Sales'].fillna('medium', inplace=True)
comp.isnull().sum()
comp.ShelveLoc.value_counts()

comp_new=pd.get_dummies(comp, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)
comp_new.columns
colnames = list(comp_new.columns)
predictors=colnames[1:11]
target=colnames[0]


# Splitting data into training and testing data set

import numpy as np

# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#data['is_train'] = np.random.uniform(0, 1, len(data))<= 0.75
#data['is_train']
#train,test = data[data['is_train'] == True],data[data['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(comp_new,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
np.mean(train.Sales == model.predict(train[predictors]))

# Accuracy = Test
np.mean(preds==test.Sales) # 1




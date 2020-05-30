# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:15:37 2020

@author: HP
"""


# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:26:36 2020

@author: HP
"""


import pandas as pd
import matplotlib.pyplot as plt
fraud = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Decision Trees\\Fraud_check.csv")
labels = ["risky", "good"]
bins=[0,30000, 100000]
fraud["Taxable"]=pd.cut(fraud['Taxable'], bins=bins, labels=labels)
fraud.head()

fraud.isnull().sum()

fraud.columns[2]
fraud.info()
fraud_new=pd.get_dummies(fraud, columns=['Undergrad','Marital','Urban'], drop_first=True)

fraud_new.columns
colnames = list(fraud_new.columns)
predictors=colnames[1:6]
target=colnames[0]


# Splitting data into training and testing data set

import numpy as np

# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#data['is_train'] = np.random.uniform(0, 1, len(data))<= 0.75
#data['is_train']
#train,test = data[data['is_train'] == True],data[data['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(fraud_new, test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
np.mean(train.Taxable == model.predict(train[predictors]))

# Accuracy = Test
np.mean(preds==test.Taxable) # 1



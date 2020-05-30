# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:49:14 2020

@author: HP
"""


import pandas as pd
import numpy as np
# Reading the Diabetes Data #################
fraud = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Random Forest\\Fraud_Check.csv")
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


X = fraud_new[predictors]
Y = fraud_new[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(fraud_new) # 600, 7 => Shape 

#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 10 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.70
rf.predict(X)
##############################

fraud_new['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Taxable']
fraud_new[cols].head()
fraud_new["Taxable"]


from sklearn.metrics import confusion_matrix
confusion_matrix(fraud_new['Taxable'],fraud_new['rf_pred']) # Confusion matrix

pd.crosstab(fraud_new['Taxable'],fraud_new['rf_pred'])



print("Accuracy",(475+115)/(475+1+9+115)*100)

# Accuracy is 98.33
fraud_new["rf_pred"]

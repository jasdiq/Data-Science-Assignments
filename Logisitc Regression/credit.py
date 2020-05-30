# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:55:38 2020

@author: HP
"""


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # train and test 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

# loading data 

credit = pd.read_csv("C:/Users/HP/Desktop/ABubakar Files/abu_Data_Science/Assignments/Logisitc Regression/creditcard.csv")
credit.tail(300)
credit.drop(credit.columns[0],inplace=True, axis=1)
# Droping first column 
#claimants.drop(["CASENUM"],inplace=True,axis = 1)

#cat_cols = ["ATTORNEY","CLMSEX","SEATBELT","CLMINSUR"]
#cont_cols = ["CLMAGE","LOSS"]

# Getting the barplot for the categorical columns 

sb.countplot(x="card",data=credit,palette="hls")
pd.crosstab(credit.card,credit.owner).plot(kind="bar")

sb.countplot(x="owner",data=credit,palette="hls")
sb.countplot(x="selfemp",data=credit,palette="hls")
sb.countplot(x="majorcards",data=credit,palette="hls")
credit.columns
pd.crosstab(credit.card,credit.selfemp).plot(kind="bar")

pd.crosstab(credit.card,credit.majorcards).plot(kind="bar")

sb.boxplot(x="card",y="age",data=credit,palette="hls")
sb.boxplot(x="card",y="income",data=credit,palette="hls")
sb.boxplot(x="card",y="share",data=credit,palette="hls")
sb.boxplot(x="card",y="expenditure",data=credit,palette="hls")

credit.isnull().sum()
credit.groupby('card').mean()

cat_vars= ["owner","selfemp","majorcards"]
subscription_dummy = pd.get_dummies(credit,columns=cat_vars)
print (len(subscription_dummy.columns))
subscription_dummy.columns.values
x= subscription_dummy.loc[:,subscription_dummy.columns!='card']
y= subscription_dummy.loc[:,subscription_dummy.columns =='card']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=0)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
rfe= RFE(logreg,10)
rfe=rfe.fit(x_train,y_train)
print(rfe.ranking_)

cols=['majorcards_0','selfemp_no','owner_no']
x=x_train[cols]
y=y_train['card']

from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)


y_pred = logreg.predict(x_test)
print (logreg.score(x_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

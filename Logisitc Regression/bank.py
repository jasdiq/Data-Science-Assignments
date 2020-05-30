# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:46:03 2020

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

# loading data file:///C:\Users\HP\Desktop\ABubakar%20Files\abu_Data_Science\Assignments\Clustering\crime_data.csv

bank = pd.read_csv("C:/Users/HP/Desktop/ABubakar Files/abu_Data_Science/Assignments/Logisitc Regression/bank-full.csv", delimiter=";",header=0)
bank.tail(300)
# Droping first column 
#claimants.drop(["CASENUM"],inplace=True,axis = 1)

#cat_cols = ["ATTORNEY","CLMSEX","SEATBELT","CLMINSUR"]
#cont_cols = ["CLMAGE","LOSS"]

# Getting the barplot for the categorical columns 

sb.countplot(x="y",data=bank,palette="hls")
pd.crosstab(bank.y,bank.job).plot(kind="bar")
bank.columns
sb.countplot(x="education",data=bank,palette="hls")
pd.crosstab(bank.y,bank.education).plot(kind="bar")
sb.countplot(x="marital",data=bank,palette="hls")
pd.crosstab(bank.y,bank.marital).plot(kind="bar")

sb.countplot(x="housing",data=bank,palette="hls")
pd.crosstab(bank.y,bank.housing).plot(kind="bar")
pd.crosstab(bank.month,bank.y).plot(kind='bar', stacked =True)
plt.title('purchase frequency for month title')
plt.xlabel('month')
plt.ylabel('frequency of purchase')
pd.crosstab(bank.loan,bank.y).plot(kind='bar',stacked =True)
plt.title('purchase frequency for loan title')
plt.xlabel('loan')
plt.ylabel('frequency of purchase')

pd.crosstab(bank.contact,bank.y).plot(kind='bar',stacked =True);plt.title('purchase frequency for contact title');plt.xlabel('contact');plt.ylabel('frequency of purchase')
cat_vars= ["job","marital","education","default","housing","loan","contact","month","poutcome"]
subscription_dummy = pd.get_dummies(bank,columns=cat_vars)
print (len(subscription_dummy.columns))
subscription_dummy.columns.values
x= subscription_dummy.loc[:,subscription_dummy.columns!='y']
y= subscription_dummy.loc[:,subscription_dummy.columns =='y']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=0)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
rfe= RFE(logreg,10)
rfe=rfe.fit(x_train,y_train)
print(rfe.ranking_)

cols=['loan_yes','job_management','contact_cellular', 'housing_yes','default_no','month_oct', 'month_sep', 'poutcome_failure', 'poutcome_other', 'poutcome_success']
x=x_train[cols]
y=y_train['y']

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

# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:13:47 2020

@author: HP
"""


import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

fire = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Support Vector Machines\\forestfires.csv")
fire.isnull().sum()
fire.head()
fire.describe()
fire["size_category"].unique()
fire.columns
fire.info()
fire['area'].describe()
labels=["min", "Avg", "max"]
bins=[0, 12, 500, 1100]
fire["area"]=pd.cut(fire['area'], bins=bins, labels=labels)
objList = fire.select_dtypes(include = "object").columns
print (objList)
#labelencoder=LabelEncoder()
le = LabelEncoder()

for feat in objList:
    fire[feat] = le.fit_transform(fire[feat].astype(str))
    print (fire.info())
    
fire.head()
fire.isnull().sum()
fire['area'].fillna('Avg', inplace=True)
fire.isnull().sum()
fire['month'].unique()
fire['day'].unique()
fire['size_category'].unique()
#fire['month_cat', 'day_cat', 'size_cat']=labelencoder.fit_transform(fire['month', 'day', 'size_category'])
fire.isnull().sum()
fire.describe().T
#fire.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
#fire.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace =True)
#fire.head()
#fire_new=pd.get_dummies(fire, columns=["size_category"])
#fire_new.head()



fire.drop(fire.columns[[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29]], axis = 1, inplace = True)
fire.head()
fire.columns

fire=fire.iloc[:,[10,0,1,2,3,4,5,6,7,8,9,11,12]]
fire.columns
fire.isnull().sum()
fire.area.value_counts()
sns.pairplot(data=fire)
sns.boxplot(x="area",y="month",data=fire,palette = "hls")
sns.boxplot(x="day",y="area",data=fire,palette = "hls")
sns.pairplot(data=fire)


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(fire,test_size = 0.3)
fire.info()
train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]
test.head()

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 59.233

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 59.61

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 59.016



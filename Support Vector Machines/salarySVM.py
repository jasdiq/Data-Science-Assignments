# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:30:42 2020

@author: HP
"""

import pandas as pd 
from glob import glob
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

stock_files= sorted(glob("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Support Vector Machines\\SalaryData_*.csv"))
salary=pd.concat((pd.read_csv(file).assign(filename=file) for file in stock_files), ignore_index=True)
salary.head()
salary.columns
salary.drop(salary.columns[[14]], axis = 1, inplace = True)
salary.columns
salary.isnull().sum()

salary.describe()
#fire["size_category"].unique()
salary.columns
salary.info()
#fire['area'].describe()
#labels=["min", "Avg", "max"]
#bins=[0, 12, 500, 1100]
#fire["area"]=pd.cut(fire['area'], bins=bins, labels=labels)
objList = salary.select_dtypes(include = "object").columns
print (objList)
#labelencoder=LabelEncoder()
le = LabelEncoder()

for feat in objList:
    salary[feat] = le.fit_transform(salary[feat].astype(str))
    print (salary.info())
    
salary.head()
salary.isnull().sum()

salary['sex'].unique()



sns.pairplot(data=salary)
sns.boxplot(x="Salary",y="age",data=salary,palette = "hls")
sns.boxplot(x="Salary",y="education",data=salary,palette = "hls")

salary=salary.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
salary.head()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(salary,test_size = 0.3)
salary.info()
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

np.mean(pred_test_linear==test_y) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) 

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)



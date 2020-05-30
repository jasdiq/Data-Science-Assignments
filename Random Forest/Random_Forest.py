import pandas as pd
import numpy as np
# Reading the Diabetes Data #################
comp = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Random Forest\\Company_Data.csv")
comp.head()
comp.columns
labels = ["low", "medium", "high"]
bins=[0,6,12,17]
comp["Sales"]=pd.cut(comp['Sales'], bins=bins, labels=labels)

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


X = comp_new[predictors]
Y = comp_new[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(comp_new) # 400, 13 => Shape 

#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 10 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.69
rf.predict(X)
##############################

comp_new['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Sales']
comp_new[cols].head()
comp_new["Sales"]


from sklearn.metrics import confusion_matrix
confusion_matrix(comp_new['Sales'],comp_new['rf_pred']) # Confusion matrix

pd.crosstab(comp_new['Sales'],comp_new['rf_pred'])



print("Accuracy",(25+129+243)/(25+2+129+1+243)*100)

# Accuracy is 99.609375
comp_new["rf_pred"]

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense #Activation,Layer,Lambda
from scipy.special import comb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

startup = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Neural Networks\\Startups.csv")

startup.head(3)
#wbcd.drop(["id"],axis=1,inplace=True) # Dropping the uncessary column 
startup.columns
startup.shape
startup.isnull().sum() # No missing values 
objList = startup.select_dtypes(include = "object").columns
print (objList)
#labelencoder=LabelEncoder()
le = LabelEncoder()

for feat in objList:
    startup[feat] = le.fit_transform(startup[feat].astype(str))
    print (startup.info())
#startup=pd.get_dummies(startup, columns=['State'], drop_first=True)
startup.columns
#startup=startup.iloc[:,[3,0,1,2,4,5]]
startup.columns
startup.shape
#  Malignant as 0 and Beningn as 1

#wbcd.loc[wbcd.diagnosis=="B","diagnosis"] = 1
#wbcd.loc[wbcd.diagnosis=="M","diagnosis"] = 0
startup.State.value_counts().plot(kind="bar")

#train,test = train_test_split(startup,test_size = 0.3,random_state=42)
#trainX = train.drop(["Profit"],axis=1)
#trainY = train["Profit"]
#testX = test.drop(["Profit"],axis=1)
#testY = test["Profit"]

########################## Neural Network for predicting continuous values ###############################

#def prep_model(hidden_dim):
 #   model = Sequential()
  #  for i in range(1,len(hidden_dim)-1):
   #        model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
     #   else:
      #      model.add(Dense(hidden_dim[i],activation="relu"))
   # model.add(Dense(hidden_dim[-1]))
    #model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    #return (model)

cont_model=Sequential()
cont_model.add(Dense(30, input_dim=4,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer="adam", metrics= ["mse"] )

column_names = list(startup.columns)
startup.columns
predictors = column_names[0:4]
target = column_names[4]
predictors.shape()
#first_model = prep_model([8,50,1])
first_model=cont_model
first_model.fit(np.array(startup[predictors]),np.array(startup[target]),epochs=10)
pred_train = first_model.predict(np.array(startup[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startup[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,startup[target],"bo")
np.corrcoef(pred_train,startup[target]) # we got high correlation 

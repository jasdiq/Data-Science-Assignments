# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:50:45 2020

@author: HP
"""


# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
dtime=pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Simple Linear Regression\\delivery_time.csv")
dtime.columns
dtime.head()
plt.hist(dtime.Delivery)
plt.boxplot(dtime.Delivery,0,"rs",0)

plt.hist(dtime.Sorting)
plt.boxplot(dtime.Sorting)

plt.plot(dtime.Delivery,dtime.Sorting,"bo");plt.xlabel("Delievry");plt.ylabel("Sorting")


dtime.Delivery.corr(dtime.Sorting) # # correlation value between X and Y
np.corrcoef(dtime.Delivery,dtime.Sorting)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Delivery~Sorting",data=dtime).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(dtime) # Predicted values of AT using the model

# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=dtime['Sorting'],y=dtime['Delivery'],color='red');plt.plot(dtime['Sorting'],pred,color='black');plt.xlabel('Sorting');plt.ylabel('Delivery')

pred.corr(dtime.Delivery) # 0.81

# Transforming variables for accuracy
model2 = smf.ols('Delivery~np.log(Sorting)',data=dtime).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(dtime['Sorting']))
pred2.corr(dtime.Delivery)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=dtime['Sorting'],y=dtime['Delivery'],color='green');plt.plot(dtime['Sorting'],pred2,color='blue');plt.xlabel('Sorting');plt.ylabel('Delivery')

# Exponential transformation
model3 = smf.ols('np.log(Delivery)~Sorting',data=dtime).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(dtime['Sorting']))
pred_log
pred3=np.exp(pred_log)
pred3
pred3.corr(dtime.Delivery)
plt.scatter(x=dtime['Sorting'],y=dtime['Delivery'],color='green');plt.plot(dtime.Sorting,np.exp(pred_log),color='blue');plt.xlabel('Sorting');plt.ylabel('Delivery')
resid_3 = pred3-dtime.Delivery
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
#student_resid = model3.resid_pearson 
#tudent_resid
#plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
#plt.scatter(x=pred3,y=wcat.AT);plt.xlabel("Predicted");plt.ylabel("Actual")



# Quadratic model
dtime["Sorting_Sq"] = dtime.Sorting*dtime.Sorting
model_quad = smf.ols("Delivery~Sorting+Sorting_Sq",data=dtime).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(dtime.Sorting)

model_quad.conf_int(0.05) # 

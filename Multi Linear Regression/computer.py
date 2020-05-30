# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:14:31 2020

@author: HP
"""
# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
comp = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Multi Linear Regression\\Computer_Data.csv")
# to get top 6 rows
comp.head(40) # to get top n rows use cars.head(10)
compdum=pd.get_dummies(comp, columns=['cd', 'multi', 'premium'], drop_first=True)
compdum.head()
compdum.columns

compdum.shape
# Correlation matrix 
compdum.corr()
comp.cd.value_counts()
comp.multi.value_counts()
comp.premium.value_counts()
comp.trend.value_counts()
# we see there exists High collinearity between input variables especially between

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(comp)


# columns names
compdum.columns


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes', data=compdum).fit()

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()
# p-values for Administration and Marketing are more than 0.05

# preparing model based only on 
ml_sp=smf.ols('price~speed',data=compdum).fit()
ml_sp.summary() #zero correlation
# p-value <0.05 .. It is significant

ml_hd=smf.ols('price~hd',data=compdum).fit()
ml_hd.summary()

ml_ram=smf.ols('price~ram',data=compdum).fit()
ml_ram.summary()

ml_screen=smf.ols('price~screen', data=compdum).fit()
ml_screen.summary()

ml_ads=smf.ols('price~ads', data=compdum).fit()
ml_ads.summary()

ml_trend=smf.ols('price~trend', data=compdum).fit()
ml_trend.summary()

ml_cd=smf.ols('price~cd_yes', data=compdum).fit()
ml_cd.summary()

ml_multi=smf.ols('price~multi_yes',data=compdum).fit()
ml_multi.summary()

ml_pre=smf.ols('price~premium_yes', data=compdum).fit()
ml_pre.summary()
# Preparing model based only on HD, RAM,speed
ml_hrs=smf.ols('price~hd+speed+ram', data=compdum).fit()
ml_hrs.summary()

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 76 AND 78 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals




# Confidence values 99%
print(ml1.conf_int(0.01)) # 99% confidence level


# Predicted values of price
profit_pred = ml1.predict(compdum[['speed','hd','ram', 'screen', 'ads', 'trend', 'cd_yes', 'multi_yes', 'premium_yes']])
profit_pred


# calculating VIF's values of independent variables
rsq_speed= smf.ols('speed~hd+ram+screen+ads+cd_yes+multi_yes+premium_yes',data=compdum).fit().rsquared  
vif_speed = 1/(1-rsq_speed) 

rsq_hd = smf.ols('hd~speed+ram+screen+ads+cd_yes+multi_yes+premium_yes',data=compdum).fit().rsquared  
vif_hd = 1/(1-rsq_hd) 

rsq_screen= smf.ols('screen~speed+ram+hd+ads+cd_yes+multi_yes+premium_yes',data=compdum).fit().rsquared  
vif_screen = 1/(1-rsq_screen) 

rsq_ram= smf.ols('ram~speed+screen+hd+ads+cd_yes+multi_yes+premium_yes',data=compdum).fit().rsquared  
vif_ram = 1/(1-rsq_ram) 

rsq_cd= smf.ols('cd_yes~speed+screen+hd+ads+ram+multi_yes+premium_yes',data=compdum).fit().rsquared  
vif_cd = 1/(1-rsq_cd) 

rsq_multi= smf.ols('multi_yes~speed+screen+hd+ads+ram+cd_yes+premium_yes',data=compdum).fit().rsquared  
vif_multi = 1/(1-rsq_multi) 
rsq_premium= smf.ols('premium_yes~speed+screen+hd+ads+ram+cd_yes+multi_yes',data=compdum).fit().rsquared  
vif_premium = 1/(1-rsq_premium) 


           # Storing vif values in a data frame
d1 = {'Variables':['speed','hd','screen','ram', 'cd_yes', 'multi_yes', 'premium_yes'],'VIF':[vif_speed,vif_hd,vif_screen,vif_ram,vif_cd,vif_multi,vif_premium]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1)

# added varible plot for weight is not showing any significance 





### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(compdum,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes', data=compdum).fit()

# train_data prediction
train_pred = model_train.predict(compdum)

# train residual values 
train_resid  = train_pred - compdum.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(compdum)

# test residual values 
test_resid  = test_pred - startup_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))



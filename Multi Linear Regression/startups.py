# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
startup = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Multi Linear Regression\\Startups.csv")

# to get top 6 rows
startup.head(40) # to get top n rows use cars.head(10)
dummies=pd.get_dummies(startup.State)
dummies.head()
merged=pd.concat([startup,dummies],axis='columns')
merged.head()
# Correlation matrix 
startup.corr()

# we see there exists High collinearity between input variables especially between
# [R&D, Marketing] , [R&D,Profit], [Marketing, profit] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup)


# columns names
startup.columns


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Profit~RD+Administration+Marketing',data=startup).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()
# p-values for Administration and Marketing are more than 0.05

# preparing model based only on 
ml_RD=smf.ols('Profit~RD',data=startup).fit()  
ml_RD.summary() # 0.947
# p-value <0.05 .. It is significant 

# Preparing model based only on 
ml_market=smf.ols('Profit~Marketing',data = startup).fit()  
ml_market.summary() #0.559

# Preparing model based only on 
ml_admin=smf.ols('Profit~Administration',data = startup).fit()  
ml_admin.summary() #0.040

# Preparing model based only on RD and Marketing
ml_rdm=smf.ols('Profit~RD+Marketing',data = startup).fit()  
ml_rdm.summary() # 0.950
# Both coefficients p-value became insignificant... 
# So there may be a chance of considering only one among VOL & WT

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 76 AND 78 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

startup_new=startup.drop(startup.index[[46,48,49,19]],axis=0)

#cars.drop(["MPG"],axis=1)

# X => A B C D 
# X.drop(["A","B"],axis=1) # Dropping columns 
# X.drop(X.index[[5,9,19]],axis=0)

#X.drop(["X1","X2"],aixs=1)
#X.drop(X.index[[0,2,3]],axis=0)


# Preparing model                  
ml_new = smf.ols('Profit~RD+Administration+Marketing',data = startup_new).fit()    

# Getting coefficients of variables        
ml_new.params

# Summary
ml_new.summary() # 0.806

# Confidence values 99%
print(ml_new.conf_int(0.01)) # 99% confidence level


# Predicted values of MPG 
profit_pred = ml_new.predict(startup_new[['RD','Marketing','Administration']])
profit_pred

startup_new.head()
# calculating VIF's values of independent variables
rsq_RD = smf.ols('RD~Marketing+Administration+Profit',data=startup_new).fit().rsquared  
vif_RD = 1/(1-rsq_RD) # 24.74

rsq_Marketing = smf.ols('Marketing~RD+Administration+Profit',data=startup_new).fit().rsquared  
vif_Marketing = 1/(1-rsq_Marketing) #3.2

rsq_admin = smf.ols('Administration~RD+Marketing+Profit',data=startup_new).fit().rsquared  
vif_admin = 1/(1-rsq_admin) #  1.23

rsq_profit = smf.ols('Profit~RD+Marketing+Administration',data=startup_new).fit().rsquared  
vif_profit = 1/(1-rsq_profit) #26

           # Storing vif values in a data frame
d1 = {'Variables':['RD','Marketing','Administration','Profit'],'VIF':[vif_RD,vif_Marketing,vif_admin,vif_profit]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)

# added varible plot for weight is not showing any significance 

# final model
final_ml= smf.ols('Profit~RD+Marketing+Administration',data=startup_new).fit()
final_ml.params
final_ml.summary() # 0.809
# As we can see that r-squared value has increased from 0.810 to 0.812.

startup_pred = final_ml.predict(startup_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(startup.Profit,startup_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
#plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
#plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(startup_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("Profit~RD+Marketing+Administration",data=startup_train).fit()

# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid  = test_pred - startup_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

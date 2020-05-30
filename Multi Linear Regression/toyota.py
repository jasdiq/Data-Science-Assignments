
# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
toyota = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Multi Linear Regression\\ToyotaCorolla.csv",  encoding = 'unicode_escape')
# to get top 6 rows
toyota.head(40) # to get top n rows use cars.head(10)
toyota.drop(toyota.columns[[0, 1, 4, 5, 7, 9, 10, 11, 14, 18, 19, 20, 21, 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]], axis = 1, inplace = True) 
# Correlation matrix 
toyota.corr()
corr = toyota.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(8, 8))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='magma', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()

# we see there exists High collinearity between input variables especially between
# [Hp & SP] , [VOL,WT] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(toyota)


# columns names
toyota.columns
toyota.shape
toyota.rename(index = {"Age_08_04": "age"},inplace = True) 
toyota.columns
# pd.tools.plotting.scatter_matrix(cars); -> also used for plotting all in one graph
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()
ml_age = smf.ols('Price~Age_08_04',data=toyota).fit()
ml_age.summary()

ml_km = smf.ols('Price~+KM',data=toyota).fit()
ml_km.summary()

ml_hp = smf.ols('Price~HP',data=toyota).fit()
ml_hp.summary()

ml_cc = smf.ols('Price~cc',data=toyota).fit()
ml_cc.summary()

ml_door = smf.ols('Price~Doors',data=toyota).fit()
ml_door.summary()

ml_gear = smf.ols('Price~Gears',data=toyota).fit()
ml_gear.summary()

ml1_tax = smf.ols('Price~Quarterly_Tax',data=toyota).fit()
ml1_tax.summary()

ml_weight = smf.ols('Price~Weight',data=toyota).fit()
ml_weight.summary()


import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 76 AND 78 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

toyota_new=toyota.drop(toyota.index[[80]],axis=0)


# Preparing model                  
ml1_new = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_new).fit()   
ml1_new.summary()
# Getting coefficients of variables        

# Confidence values 99%
print(ml1_new.conf_int(0.01)) # 99% confidence level


# Predicted values of MPG 
price_pred = ml1_new.predict(toyota_new[['Age_08_04','KM', 'HP', 'cc','Doors','Gears','Quarterly_Tax','Weight']])
price_pred

toyota_new.head()
# calculating VIF's values of independent variables
rsq_age = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_new).fit().rsquared  
vif_age = 1/(1-rsq_age)

rsq_km = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_new).fit().rsquared  
vif_km = 1/(1-rsq_km)

rsq_hp = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_new).fit().rsquared  
vif_hp = 1/(1-rsq_hp)

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=toyota_new).fit().rsquared  
vif_cc = 1/(1-rsq_cc)

rsq_door = smf.ols('Doors ~ cc+Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight',data=toyota_new).fit().rsquared  
vif_door = 1/(1-rsq_door)
rsq_gear = smf.ols('Gears~cc+Age_08_04+KM+HP+Doors+Quarterly_Tax+Weight',data=toyota_new).fit().rsquared  
vif_gear = 1/(1-rsq_gear)

rsq_tax = smf.ols('Quarterly_Tax~Gears+cc+Age_08_04+KM+HP+Doors+Weight',data=toyota_new).fit().rsquared  
vif_tax = 1/(1-rsq_tax)

rsq_weight = smf.ols('Weight~Gears+cc+Age_08_04+KM+HP+Doors+Quarterly_Tax',data=toyota_new).fit().rsquared  
vif_weight = 1/(1-rsq_weight)




           # Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM', 'HP', 'cc','Doors','Gears','Quarterly_Tax','Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_door,vif_gear,vif_tax,vif_weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1_new)


### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
toyota_train,toyota_test  = train_test_split(toyota_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=toyota_train).fit()

# train_data prediction
train_pred = model_train.predict(toyota_train)

# train residual values 
train_resid  = train_pred - toyota_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(toyota_test)

# test residual values 
test_resid  = test_pred - toyota_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

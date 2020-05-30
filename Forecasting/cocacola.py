# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:09:41 2020

@author: HP
"""



import pandas as pd
coca = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\Assignments\\Forecasting\\CocaCola_Sales_Rawdata.csv")
quarter =["Q1", "Q2", "Q3", "Q4"] 
import numpy as np
p = coca["Quarter"][0]
p[0:3]
coca['quarter']= 0

for i in range(42):
    p = coca["Quarter"][i]
    coca['quarter'][i]= p[0:3]
coca.head()
    
q_dummies = pd.DataFrame(pd.get_dummies(coca['quarter']))
coca1 = pd.concat([coca,q_dummies],axis = 1)

coca1["t"] = np.arange(1,43)

coca1["t_squared"] = coca1["t"]*coca1["t"]
coca1.columns
coca1["log_sales"] = np.log(coca1["Sales"])
#airline1.rename(columns={"Ridership ": 'Ridership'}, inplace=True)
coca1.Sales.plot()
Train = coca1.head(30)
Test = coca1.tail(13)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1_+Q2_+Q3_+Q4_',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1_','Q2_','Q3_','Q4_']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1_+Q2_+Q3_+Q4_',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1_','Q2_','Q3_','Q4_','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Q1_+Q2_+Q3_+Q4_',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Q1_+Q2_+Q3_+Q4_',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 


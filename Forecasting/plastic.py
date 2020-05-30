# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:02:21 2020

@author: HP
"""

import pandas as pd
plastic = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\Assignments\\Forecasting\\PlasticSales.csv", index_col=[0], squeeze=True)
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
plastic.plot()
plastic.plot(style='k.')
plastic.size
plastic.describe()
#moving average
plastic_ma=plastic.rolling(window=20).mean()
plastic_ma.plot()
#lag or shift method
plastic_base=pd.concat([plastic, plastic.shift(1)], axis=1)
plastic_base.columns=["Actual_sales", "Forecast_sales"]
plastic_base.dropna(inplace=True)
from sklearn.metrics import mean_squared_error
import numpy as np
plastic_error=mean_squared_error(plastic_base.Actual_sales, plastic_base.Forecast_sales)
np.sqrt(plastic_error)

########ARIMA Auto regressive method######
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(plastic)
#q-2
plot_pacf(plastic)
#p-1
#d 0-1
from statsmodels.tsa.arima_model import ARIMA
plastic_train = plastic[0:50]
plastic_test = plastic[50:60]
plastic_model =ARIMA(plastic_train, order=(1,2,2))
plastic_model_fit=plastic_model.fit()
plastic_model_fit.aic
plastic_forecast=plastic_model_fit.forecast(steps=10)[0]
np.sqrt(mean_squared_error(plastic_test, plastic_forecast))
p_values = range(0,5)
d_values = range(0,3)
q_values = range(0,5)
import warnings
warnings.filterwarnings("ignore")
for p in p_values:
    for d in d_values:
        for q in q_values:
            order=(p,d,q)
            train,test=plastic[0:50], plastic[50:60]
            predictions =list()
            for i in range(len(test)):
                try:
                    model=ARIMA(order)
                    model_fit=model.fit(disp=0)
                    pred_y=model.forecast()[0]
                    predictions.append(pred_y)
                    error=mean_squared_error(test, predictions)
                    print('ARIMA%s  RMSE= %2f'% (order,error))
                except:
                    continue

#ARIMA(p,d,q)          ###only p term need to specify in Auto regressive model

#ARIMA(2,0,0)
#ARIMA(0,0,2)        ##only q term need to specify in moving avearage model

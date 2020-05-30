
# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
empd=pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Simple Linear Regression\\emp_data.csv")
empd.columns
empd.head()
plt.hist(empd.Churn)
plt.boxplot(empd.Churn,0,"rs",0)

plt.hist(empd.Salary)
plt.boxplot(empd.Salary,0,"rs",0)

plt.plot(empd.Churn,empd.Salary,"bo");plt.xlabel("Churn");plt.ylabel("Salary")


empd.Churn.corr(empd.Salary) # # correlation value between X and Y
np.corrcoef(empd.Churn,empd.Salary)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Churn~Salary",data=empd).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(empd) # Predicted values  using the model
pred
# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=empd['Salary'],y=empd['Churn'],color='red');plt.plot(empd['Salary'],pred,color='black');plt.xlabel('Salary');plt.ylabel('Churn')

pred.corr(empd.Salary) 

# Transforming variables for accuracy
model2 = smf.ols('Churn~np.log(Salary)',data=empd).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(empd['Salary']))
pred2.corr(empd.Churn)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=empd['Salary'],y=empd['Churn'],color='green');plt.plot(empd['Salary'],pred2,color='blue');plt.xlabel('Salary');plt.ylabel('Churn')
# Exponential transformation
model3 = smf.ols('np.log(Churn)~Salary',data=empd).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(empd['Salary']))
pred_log
pred3=np.exp(pred_log)
pred3
pred3.corr(empd.Churn)
plt.scatter(x=empd['Salary'],y=empd['Churn'],color='green');plt.plot(empd.Salary,np.exp(pred_log),color='blue');plt.xlabel('Salary');plt.ylabel('Churn')
#resid_3 = pred3-dtime.Delivery
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
#student_resid = model3.resid_pearson 
#tudent_resid
#plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
#plt.scatter(x=pred3,y=wcat.AT);plt.xlabel("Predicted");plt.ylabel("Actual")



# Quadratic model
empd["Salary_Sq"] = empd.Salary*empd.Salary
model_quad = smf.ols("Churn~Salary+Salary_Sq",data=empd).fit()
model_quad.params
model_quad.summary()
#pred_quad = model_quad.predict(dtime.Sorting)

#model_quad.conf_int(0.05) # 



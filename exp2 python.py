#Program to implement the simple linear regression model for predicting the marks scored.
#Developed by: PONGURU AASRITH SAIRAM
#RegisterNumber: 212223240116


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('student_scores.csv')


data.head()
print("Data Head :\n" ,data.head())
data.tail()
print("\nData Tail :\n" ,data.tail())


x=data.iloc[:,:-1].values  
y=data.iloc[:,1].values

print("\nArray value of X:\n" ,x)
print("\nArray value of Y:\n", y)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0 )

regressor=LinearRegression() 
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test) 

print("\nValues of Y prediction :\n",y_pred)

print("\nArray values of Y test:\n",y_test)


print("\nTraining Set Graph:\n")
plt.scatter(x_train,y_train,color='red') 
plt.plot(x_train,regressor.predict(x_train),color='green') 
plt.title("Hours Vs Score(Training set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

y_pred=regressor.predict(x_test) 

print("\nTest Set Graph:\n")
plt.scatter(x_test,y_test,color='red') 
plt.plot(x_test,regressor.predict(x_test),color='green') 
plt.title("Hours Vs Score(Test set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(x, y)
mse = metrics.mean_squared_error(x, y)
rmse = np.sqrt(mse)  

print("\n\nValues of MSE, MAE and RMSE : \n")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)

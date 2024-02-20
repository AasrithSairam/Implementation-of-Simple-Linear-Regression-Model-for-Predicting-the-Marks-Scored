# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use Python's standard libraries.
2.Set variables to assign dataset values.
3.Import LinearRegression from sklearn.
4.Assign points to represent the graph.
5.Predict the regression for marks based on the graph representation.
6.Compare the graphs to determine the LinearRegression for the provided data.

## Program:
```
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

```

## Output:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/a12f3fd3-cf5f-45d0-be29-c1e080dd9fcc)
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/19487ed6-6e1a-4ccc-b750-8f11f4323853)
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/ac085655-537c-46a5-b668-5c8d9c441f8d)
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/360c861f-e3f1-41d0-ba5c-697b695be16d)
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/6d261d63-77d3-47d8-af55-640eb2ffba25)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

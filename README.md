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
Data Head:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/caa82629-7f93-4b8f-9bd3-d26952eaf2ed)
Data Tail:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/b2fadae5-99a4-4ae8-8359-0543af5051da)
Array Value of X:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/beab0dad-5f1e-4fe7-b74a-f87de73df7e5)
Array Value of Y:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/413fb93c-5463-4f57-9a23-60f975611431)
Values of Y prediction:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/bc0be980-534f-4334-8393-12e09ee556e5)
Array Values of Y Test:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/5b261c40-493a-4080-a4a3-0965e0e7f577)
Training Set Graph:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/8ba6c8f3-55ee-45f4-b952-13e00129b923)
Test Set Graph:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/64067fec-6e0a-41d7-bf8d-86fd45b2a8d4)
Values of MSE, MAE and RMSE:
![image](https://github.com/AasrithSairam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331438/229facb8-feb6-45e4-9120-a82706b65257)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

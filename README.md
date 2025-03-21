# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
```
## Program and Output
```
Developed by: Janani s
RegisterNumber:  212224230103
```
```
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)

        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/84eecd5e-e56e-42ae-aacc-4ab909ad01e7)

```
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
```
![image](https://github.com/user-attachments/assets/c93ea9ca-15b5-47ae-a714-76dd41c62e2d)



```
print(X1_Scaled)
```
![image](https://github.com/user-attachments/assets/96c80372-b748-43b8-9423-ac28bfd25cd4)

```
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
![image](https://github.com/user-attachments/assets/05c88746-949b-47ad-a309-2b21bec6d8d4)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

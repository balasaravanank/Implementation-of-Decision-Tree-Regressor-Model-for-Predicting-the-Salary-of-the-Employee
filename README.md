# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries.
2.Load the dataset using pandas.
3.Check for null values.
4.Encode categorical column (Position) using LabelEncoder.
5.Split the data into features (x) and target (y).
6.Split into training and testing sets.
7.Train the DecisionTreeRegressor on training data.
8.Predict on test data.
9.Evaluate model using MSE and R² Score.
10.Predict on a new sample [5, 6].
11.Visualize the decision tree.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Load dataset and check for nulls
data = pd.read_csv("Salary_EX7.csv")
print(data.isnull().sum())

#Encode 'Position'
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

#Train Decision Tree Regressor
x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

#Predict test data
y_pred = dt.predict(x_test)

#Calculate MSE and R2
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

#Predict from array
print("Prediction for [5,6]:", dt.predict([[5, 6]]))

#Visualize the tree
plt.figure(figsize=(15, 6))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()


```

## Output:


## Developed by : BALA SARAVANAN K
## Reg no: 24900611
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

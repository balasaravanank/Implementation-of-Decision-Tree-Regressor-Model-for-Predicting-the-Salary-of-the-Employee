# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries

2.Load and check dataset

3.Encode categorical data

4.Split into features and target

5.Train Decision Tree Regressor

6.Predict and evaluate

7.Predict new value

8.Visualize tree

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
## FIRST 5 ROWS OF THE DATASET
![Screenshot 2025-04-21 220119](https://github.com/user-attachments/assets/e6256b17-d0a1-4d72-9566-265aa97e33b2)
## DATASET INFORMATION (DATA TYPES AND NON-NULL COUNTS)
![Screenshot 2025-04-21 220138](https://github.com/user-attachments/assets/f58cdce3-d0a6-4fbb-a121-ab7db26c6710)
## DECISION TREE REGRESSOR
![Screenshot 2025-04-21 220204](https://github.com/user-attachments/assets/1dc8ca35-9b8c-426c-8acc-e9b80091a4e8)
## MEAN SQUARED ERROR (MSE) AND R2 SCORE
![Screenshot 2025-04-21 220208](https://github.com/user-attachments/assets/e6bebd45-5e3e-48b5-aa07-426cf43ddd6f)
## PREDICTED SALARY FOR INPUT [5, 6]
![Screenshot 2025-04-21 220214](https://github.com/user-attachments/assets/93fe146a-ebc9-48ea-b16d-2c45e2e81009)
## DECISION TREE VISUALIZATION
![Screenshot 2025-04-21 220224](https://github.com/user-attachments/assets/f2b5cf44-6267-4ddf-8026-e708a2a6725e)


## Developed by : BALA SARAVANAN K
## Reg no: 24900611
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

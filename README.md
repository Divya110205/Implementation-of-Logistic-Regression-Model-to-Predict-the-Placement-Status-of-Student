# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. .Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DIVYA.A
RegisterNumber:  212222230034

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![1](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/beaaf4cc-7814-4986-994a-0bb739596fc2)

![2](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/24c3b1b1-2ab5-4dbb-ae00-91985f64adfb)

![3](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/4aa2e961-34ca-4756-bd40-6491a00c4b2d)

![4](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/83829f23-3941-4eef-a6a4-c7346096ec1e)

![5](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/7f50391d-c4bf-4a6e-895b-f0db8500dd8c)

![6](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/083cf49c-fcef-4134-bc36-701e3c69a344)

![7](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/ebc079ff-a816-4216-854a-842a13bf87e2)

![8](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/11d132c8-1bf7-4438-870b-b81506b77d88)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

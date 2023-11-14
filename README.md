# EX 4-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## DATE: 19.09.23
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

# Placement Data:
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

# Salary Data:
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

# Checking The null() Function:
data1.isnull().sum()

# Data Duplicate:
data1.duplicated().sum()

# Print Data:
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

# Data-status:
x=data1.iloc[:,:-1]
x

y=data1["status"]
y

# y_prediction Array:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

# Accuracy Value:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

# Confusion Array:
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

# Classification Report:
from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

# Prediction Of LR:
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output: 
### Placement Data:
![1](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/1581e3a6-91fc-485d-ad2c-7db31dea3911)

### Salary Data:
![2](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/cecca214-d654-4f2e-a961-68ec8fbd4f78)

### Checking The null() Function:
![3](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/3c44aff4-1a5b-4c93-bc78-345b16407d5a)

### Data Duplicate:
![4](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/ade45585-91bf-4e3f-bc8e-02f8d97aa786)

### Print Data:
![5](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/cc8a00dd-74be-494b-8581-203cf2a01b1f)

### Data-status:
![6(1)](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/5c8af787-a5c7-423e-8230-2ded03ec5a70)
![6(2)](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/6709c0e8-a370-42c0-bb8a-47d7d9097305)

### y_prediction Array:
![7](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/a4e07ef5-9848-4e81-83f3-f587f11d0087)

### Accuracy Value:
![8](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/68a64017-db66-4f34-91bb-5397ef44e889)

### Confusion Array:
![9](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/39104fc5-8391-4498-b2d8-1289a474534b)

### Classification Report:
![10](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/8f467b39-3edf-4811-bbc6-5000334f0284)

### Prediction Of LR:
![11](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/f45c80d8-a4f5-4a40-9e97-d51fd2d2eda9)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

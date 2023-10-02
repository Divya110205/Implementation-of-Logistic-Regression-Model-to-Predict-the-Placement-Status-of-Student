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
![1](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/97400ecc-7ef7-4ee8-a384-85228269522f)

### Salary Data:
![2](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/09b61839-78fc-4cdd-9a1d-fd77712ba852)

### Checking The null() Function:
![3](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/26fffd09-ab35-4ee6-bfd7-45760437e145)

### Data Duplicate:
![4](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/fafbeb4f-2fd8-494a-b7d4-3039d7823f76)

### Print Data:
![5](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/56f72605-bfc1-4aad-866a-ba2f280247b7)

### Data-status:
![6(1)](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/5ee55782-bec1-4806-97d2-cdec183831a7)
![6(2)](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/d6b348d2-3ca7-4586-81f9-e59ddd674cff)

### y_prediction Array:
![7](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/741ff392-73da-48f3-9b5e-e1b65b8ec4f8)

### Accuracy Value:
![8](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/90caf5a2-e040-4c6f-8bea-b09217768995)

### Confusion Array:
![9](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/26bb0778-96e8-418f-a937-fd9d86178171)

### Classification Report:
![10](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/e268c9dc-7a90-4dc4-8057-1fd57a490755)

### Prediction Of LR:
![11](https://github.com/Divya110205/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404855/0cea0f2c-f731-4eb8-be33-6d841ef4175f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

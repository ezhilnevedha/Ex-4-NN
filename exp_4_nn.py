#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

irisdata = pd.read_csv(url, names=names)

#Takes first 4 columns and assign them to variable "X"

X = irisdata.iloc[:, 0:4]

#Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.

y = irisdata.select_dtypes(include=[object])

X.head()

y.head()

#y actually contains all categories or classes:

y.Class.unique()

#Now transforming categorial into numerical values

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)

y.head()

#Now for train and test split (80% of  dataset into  training set and  other 20% into test data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Feature scaling

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)

print(predictions)

#Last thing: evaluation of algorithm performance in classifying flowers

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

arr = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

df = pd.read_csv(url, names=arr)

print(df.head())

a = df.iloc[:, 0:4]

b = df.select_dtypes(include=[object])

b = df.iloc[:,4:5]

training_a, testing_a, training_b, testing_b = train_test_split(a, b, test_size = 0.25)

myscaler = StandardScaler()

myscaler.fit(training_a)

training_a = myscaler.transform(training_a)

testing_a = myscaler.transform(testing_a)

m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500)

m1.fit(training_a, training_b.values.ravel())


predicted_values = m1.predict(testing_a)
print(confusion_matrix(testing_b,predicted_values))
print(classification_report(testing_b,predicted_values))


# In[ ]:





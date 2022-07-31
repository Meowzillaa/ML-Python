import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]
my_data.shape
print(my_data)

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

y = my_data["Drug"] #Fill the target variable.
y[0:5]

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Set Shapes
print('Shape of x testset', X_testset.shape, '&', 'Shape of y testset', y_testset.shape)
print('Shape of x trainset', X_trainset.shape, '&', 'Shape of y trainset', y_trainset.shape)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # shows the default parameters
drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("Decision Tree Accuracy is", metrics.accuracy_score(y_testset, predTree))

tree.plot_tree(drugTree)
plt.show()
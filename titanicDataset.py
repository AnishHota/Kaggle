#took help from datacamp
import pandas as pd
import numpy as np

#machine learning
from sklearn.tree import DecisionTreeClassifier
#converting the csv files to dataframes using panda
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#cleaning the data using panda DataFrame
train['Sex'][train['Sex']=='male'] = 0
train['Sex'][train['Sex']=='female'] = 1
test['Sex'][test['Sex']=='male'] = 0
test['Sex'][test['Sex']=='female'] = 1
#filling the nan values with their medians
train['Age']=train['Age'].fillna(train['Age'].median())
test['Age']=test['Age'].fillna(test['Age']).median()
#extracting the target,train_features,test_features
target = train['Survived'].values
train_features = train[['Age','Sex','Parch']].values
test_features = test[['Age','Sex','Parch']].values
#making the classifier
clf = DecisionTreeClassifier()
#fitting the classifier
clf.fit(train_features,target)
#predicting
pred = clf.predict(test_features)
print clf.feature_importances_
#using numpy converting the PassengerId as an array of type int
PassengerId = np.array(test['PassengerId']).astype(int)
#converting the numpy to dataframe using pd and the predictions and making columns of survived
my_solution1 = pd.DataFrame(pred,PassengerId,columns=['Survived'])
print(my_solution1)
#.shape prints the no. of rows in the dataframe
print(my_solution1.shape)
#converting the dataframe to csv so that it can be submitted to kaggle
my_solution1.to_csv('solution1.csv',index_label=['PassengerId'])
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(train_features,target)
pred = clf.predict(test_features)
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution2 = pd.DataFrame(pred,PassengerId,columns=['Survived'])
my_solution2.to_csv('my_solution2.csv',index_label=['PassengerId'])

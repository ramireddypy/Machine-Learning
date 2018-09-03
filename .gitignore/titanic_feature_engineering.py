
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree, ensemble, preprocessing, model_selection, feature_selection

import os
os.chdir("../input")

train = pd.read_csv(r'train.csv')
test = pd.read_csv(r'test.csv')

train.Age = preprocessing.Imputer().fit_transform(train[['Age']])
test.Age = preprocessing.Imputer().fit_transform(test[['Age']])
test.Fare = preprocessing.Imputer().fit_transform(test[['Fare']])
train.info()
combo = pd.concat([train,test], axis=0, sort=False)
combo.info()

combo.loc[combo['Embarked'].isna(), 'Embarked'] = combo.Embarked.mode()
combo['FamilyCount'] = combo['SibSp'] + combo['Parch'] + 1
combo['FamilySize'] = combo['FamilyCount'].apply(lambda val : 'Single' if val < 2 else 'Small' if val < 4 else 'large' if val < 5 else 'Big')
combo['AgeType'] = combo['Age'].apply(lambda val : 'Child' if val < 12 else 'Teen' if val <= 21 else 'Young' if val < 30 else 'Middle' if val < 45 else 'Senior citizen' if val < 60 else 'Old')
combo['Title'] = combo['Name'].apply(lambda name : name.split(',')[1].split('.')[0])

train = combo[:891]
test = combo[891:]
test.drop(columns=['Survived'], axis=1, inplace=True)
X_train = pd.get_dummies(data = train.drop(columns=['Name','PassengerId','SibSp','Parch','Age','Fare','Cabin', 'Ticket', 'Survived'], axis=1))
y_train = train.loc[:'Survived']
X_test = pd.get_dummies(data = train.drop(columns=['Name','PassengerId','SibSp','Parch','Fare','Age','Cabin', 'Ticket'], axis=1))

classifer = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[3,4,5,6], 'criterion':['gini','entropy']}
grid_classifier = model_selection.GridSearchCV(classifer, param_grid=dt_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_
X_train.info()


classifer = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[3,4,5,6], 'criterion':['gini','entropy']}
grid_classifier = model_selection.GridSearchCV(classifer, dt_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_
X_train.info()


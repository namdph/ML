import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('1-training-data.csv')
hw_data = pd.read_csv('20170218-test.csv', names=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'y'])

train_data = data.copy()
for column in train_data:
    train_data[column] = pd.to_numeric(train_data[column], errors='coerce')
train_data.fillna(train_data.mean(), inplace=True)

features = np.array(train_data[train_data.columns[:-1]])
labels = np.array(train_data[train_data.columns[-1]])
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# create new a random forest model
rf = RandomForestClassifier(random_state=42)
# create a dictionary of all values we want to test for n_estimators
param_grid = {"n_estimators": np.arange(100, 500)}
# use gridsearch to test all values for n_estimators
rf_grid = GridSearchCV(rf, param_grid=param_grid, cv=5)
# fit model to data
rf_grid.fit(train_features, train_labels)
# check top performing n_estimators value
print("Best RF Model Parameter: ", rf_grid.best_params_)
# train random forest model using top performing n_estimators value
rf_model = RandomForestClassifier(n_estimators=144, criterion='gini',bootstrap=True)
rf_model.fit(train_features, train_labels)
rf_prediction = rf_model.predict(test_features)
# check accuracy
print("RF Accuracy: ", metrics.accuracy_score(test_labels, rf_prediction))

# create new a knn model
knn2 = KNeighborsClassifier()
# create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
# fit model to data
knn_gscv.fit(train_features, train_labels)
knn_gscv.predict(test_features)
# check top performing n_neighbors value
print("Best KNN Model Parameter: ", knn_gscv.best_params_)
# train KNN model using top performing n_neighbors value
knn = KNeighborsClassifier(n_neighbors=1).fit(train_features, train_labels)
knn_prediction = knn.predict(test_features)
# check accuracy
print("KNN Accuracy: ", metrics.accuracy_score(test_labels, knn_prediction))

# Prediction of need-to-test file
hw_features = np.array(hw_data[hw_data.columns[:-1]])
hw_labels = np.array(hw_data[hw_data.columns[-1]])
hw_prediction = rf_model.predict(hw_features)
hw_prediction2 = knn.predict(hw_features)
print("Given results:", hw_labels)
print("RF Prediction:", hw_prediction)
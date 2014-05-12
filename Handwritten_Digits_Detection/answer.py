__author__ = 'Katherine'

import csv
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import argparse
from sklearn import cross_validation
from sklearn import metrics
import collections


# Part 1: Load the data from the csv file
X = []
Y = []
with open('/Users/Katherine/PycharmProjects/630TM_HW1/letters_training.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        i += 1
        if i==1:
            n_features = len(row)-1
        if row[0:n_features].count('') == 0:
            try:
                X.append(row[0:n_features])
                Y.append(row[n_features])
            except ValueError:
                break
X = np.array(X, dtype=int)
Y = np.array(Y, dtype=int)


# Part 2: evaluate the classifier performance
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.15, random_state=0)
# Train classifier
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=y_train, n_folds=3)
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid_linear = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv)
grid_rbf.fit(X_train, y_train)
grid_linear.fit(X_train, y_train)
# Use the best estimator to train the training data
rbf_svc = grid_rbf.best_estimator_.fit(X_train, y_train)
linear_svc = grid_linear.best_estimator_.fit(X_train, y_train)
# Evaluation the tuned classifier on the remaining 15% hold-out set
print "Score of SVM with RBF kernel: ", rbf_svc.score(X_test, y_test)
print "score of SVM with Linear kernel: ",linear_svc.score(X_test, y_test)
if rbf_svc.score(X_test, y_test) > linear_svc.score(X_test, y_test):
    print "SVM with RBF kernel has a better performance.\n"
elif rbf_svc.score(X_test, y_test) < linear_svc.score(X_test, y_test):
    print "SVM with Linear kernel has a better performance.\n"
else:
    print "SVM with RBF kernel and Linear kernel have same performance.\n"

# Part 3: Check whether the classifier predicts a handwritten '9'
# Read the file 9_54.txt and convert the row-major ordered matrix into a one-dimensional array
Z = []
with open('/Users/Katherine/PycharmProjects/630TM_HW1/9_54.txt', 'r') as f:
    lines = f.read()
    Z.append(lines.replace('\r\n', ''))
Z = np.array(map(int, ','.join(Z[0]).split(',')))
# Check that your classifier predicts that this file represents a handwritten '9'
if rbf_svc.predict(Z) == -1:
    print 'The classifier predicts that the file represents a handwritten \'9\'. \n'

# Part 4: Reduce the dimensional of feature vectors and reevaluate the performance
# Remove the features are constant over the training set
index = []
for i in range(len(X_train)):
    if np.sum(X_train[:, i]) != 0 and np.sum(X_train[:, i]) != len(X_train[i]):
        index.append(i)
index = np.array(index)
X_train_new = np.delete(X_train, index, axis=1)
X_test_new = np.delete(X_test, index, axis=1)
# Evaluation the tuned classifier on the remaining 15% hold-out set
rbf_svc_new = grid_rbf.best_estimator_.fit(X_train_new, y_train)
print "Score of SVM with RBF kernel after reducing the dimension of feature vectors: "\
        , rbf_svc_new.score(X_test_new, y_test), "\n"





__author__ = 'Katherine'

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import re
import csv
import datetime
import numpy as np
import sys
import argparse


def is_float(str):
    """
    Check whether a string is a float.
    """
    try:
        float(str)
        return True
    except ValueError:
        return False


def is_int(str):
    """
    Check whether a string is a integer.
    """
    try:
        int(str)
        return True
    except ValueError:
        return False


def cvrt_to_num_if_can(str):
    """
    If string is really a number, convert it to integer or float.
    """
    if is_int(str):
        return int(str)
    elif is_float(str):
        return float(str)
    else:
        return str


def readCSV(filename):
    """
    Read in training and test data sets.
    """
    dic = {
        'target': [],
        'target_names': [],
        'data': [],
        'feature_names': []
    }

    input_file = open(filename, 'r')
    lines = input_file.readlines()
    input_file.close()
    headers = re.sub("\"", "", lines.pop(0)).strip().split(',')

    dic['feature_names'] = headers[1:]

    for line in lines:
        data = re.sub("\"", "", line).strip().split(',')
        data = [cvrt_to_num_if_can(d) for d in data]
        # print data
        target = data.pop(0)
        dic['target'].append(target)
        if str(target) not in dic['target_names']:
            dic['target_names'].append(target)
        dic['data'].append(data)
    dic['target'] = np.array(dic['target'])
    dic['target_names'] = np.unique(dic['target_names'])
    dic['data'] = np.array(dic['data'])
    dic['feature_names'] = np.array(dic['feature_names'])
    return dic


def modelSelection(x_train, y_train, x_test, y_test, model, n_folds):
    """
    Select various models and return the AUCs of training and test sets and predicted offer acceptance probabilities.
    """
    if model == "Random Forest":
        clf = RandomForestClassifier(n_estimators=150, oob_score=True, random_state=0, min_samples_split=1)
    elif model == "Logistic Regression L1":
        clf = LogisticRegression(penalty='l1', random_state=0, class_weight='auto')
    elif model == "Logistic Regression L2":
        clf = LogisticRegression(penalty='l2', random_state=0, class_weight='auto')
    elif model == "Decision Tree":
        clf = DecisionTreeClassifier(random_state=0)
    elif model == "Naive Bayes":
        clf = GaussianNB()
    elif model == "KNN":
        clf = KNeighborsClassifier(n_neighbors=10)
    # Perform cross-validation on training dataset and calculate AUC
    cv = StratifiedKFold(y_train, n_folds=n_folds)
    auc_train = []
    auc_validation = []
    auc_test = []
    pred_prob = []
    for i, (train, validation) in enumerate(cv):
        clf = clf.fit(x_train[train], y_train[train])
        auc_train.append(metrics.roc_auc_score(y_train[train], clf.predict_proba(x_train[train])[:, 1]))
        auc_validation.append(metrics.roc_auc_score(y_train[validation], clf.predict_proba(x_train[validation])[:, 1]))
        auc_test.append(metrics.roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
        pred_prob.append(clf.predict_proba(x_test)[:, 1])
    return np.mean(auc_train), np.mean(auc_validation), np.mean(auc_test), np.mean(pred_prob, axis=0)



if __name__ == "__main__":
    # Read the csv file, and convert it into the structured dict.
    print datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', required=True)
    parser.add_argument('-f2', required=True)
    args = parser.parse_args()
    trainData = readCSV(args.f1)
    testData = readCSV(args.f2)
    print "finish loading data...\n"

    x_train = trainData['data']
    y_train = trainData['target']
    x_test = testData['data']
    y_test = testData['target']
    print datetime.datetime.now()
    print "start building model...outputting the AUCs...\n"

    # Random Forest.
    auc_train_rf, auc_validation_rf, auc_test_rf, prob1_rf= \
        modelSelection(x_train, y_train, x_test, y_test, "Random Forest", 7)
    print "AUC of training (Random Forest): %f" % auc_train_rf
    print "AUC of validation (Random Forest): %f" % auc_validation_rf
    print "AUC of test (Random Forest): %f" % auc_test_rf, "\n"

    # Logistic Regression (L1).
    auc_train_lr1, auc_validation_lr1, auc_test_lr1, prob1_lr1 = \
        modelSelection(x_train, y_train, x_test, y_test, "Logistic Regression L1", 7)
    print "AUC of training (Logistic Regression with L1): %f" % auc_train_lr1
    print "AUC of validation (Logistic Regression with L1): %f" % auc_validation_lr1
    print "AUC of test (Logistic Regression with L1): %f" % auc_test_lr1, "\n"

    # Logistic Regression (L2).
    auc_train_lr2, auc_validation_lr2, auc_test_lr2, prob1_lr2 = \
        modelSelection(x_train, y_train, x_test, y_test, "Logistic Regression L2", 7)
    print "AUC of training (Logistic Regression with L2): %f" % auc_train_lr2
    print "AUC of validation (Logistic Regression with L2): %f" % auc_validation_lr2
    print "AUC of test (Logistic Regression with L2): %f" % auc_test_lr2, "\n"

    # Decision Tree.
    auc_train_dt, auc_validation_dt, auc_test_dt, prob1_dt = \
        modelSelection(x_train, y_train, x_test, y_test, "Decision Tree", 7)
    print "AUC of training (Decision Tree): %f" % auc_train_dt
    print "AUC of validation (Decision Tree): %f" % auc_validation_dt
    print "AUC of test (Decision Tree): %f" % auc_test_dt, "\n"

    # Gaussian Naive Bayes.
    auc_train_nb, auc_validation_nb, auc_test_nb, prob1_nb = \
        modelSelection(x_train, y_train, x_test, y_test, "Naive Bayes", 7)
    print "AUC of training (Gaussian Naive Bayes): %f" % auc_train_nb
    print "AUC of validation (Gaussian Naive Bayes): %f" % auc_validation_nb
    print "AUC of test (Gaussian Naive Bayes): %f" % auc_test_nb, "\n"

    # K-nearest Neighbors.
    auc_train_knn, auc_validation_knn, auc_test_knn, prob1_knn = \
        modelSelection(x_train, y_train, x_test, y_test, "KNN", 7)
    print "AUC of training (K-nearest Neighbors): %f" % auc_train_knn
    print "AUC of validation (K-nearest Neighbors): %f" % auc_validation_knn
    print "AUC of test (K-nearest Neighbors): %f" % auc_test_knn, "\n"
    print datetime.datetime.now()
    print "finish building model...writing csv file...\n"

    # Write output into a csv file
    with open('part1_results.csv', 'wb') as f:
        a = csv.writer(f, delimiter=',')
        observation_id = np.array(range(len(testData['data'])))
        observation_id = np.insert(observation_id.astype('str'), 0, "Observation ID").reshape((len(testData['data'])+1, 1))
        data_lr1 = np.insert(prob1_lr1.astype('str'), 0, "Logistic Regression (L1)").reshape((len(testData['data'])+1, 1))
        data_lr2 = np.insert(prob1_lr2.astype('str'), 0, "Logistic Regression (L2)").reshape((len(testData['data'])+1, 1))
        data_rf = np.insert(prob1_rf.astype('str'), 0, "Random Forest").reshape((len(testData['data'])+1, 1))
        data_dt = np.insert(prob1_dt.astype('str'), 0, "Decision Tree").reshape((len(testData['data'])+1, 1))
        data_knn = np.insert(prob1_knn.astype('str'), 0, "K-nearest Neighbors").reshape((len(testData['data'])+1, 1))
        a.writerows(np.hstack((observation_id, data_lr1, data_lr2, data_rf, data_dt, data_knn)))
    sys.stdout.flush()
    f.close()

    print datetime.datetime.now()
    print "part 1 is complete...\n"

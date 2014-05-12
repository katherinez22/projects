__author__ = 'Katherine'

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
import argparse
import datetime
import numpy as np
import operator


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


def profit(prob1, amount, rate, term):
    """
    Compute the expected profit from each customer.
    """
    profit = np.array(prob1) * np.array(amount) * (np.array(rate)/100-0.025) * np.array(term) / 12
    return np.mean(profit)


def rateFinder(x_opt, clf):
    """
    Find a single rate that maximizes the expected profit.
    """
    profitList = []  # store expected profit for each customer
    prob = []  # store probability of acceptance with various rate for each customer
    rateList = np.linspace(3, 14, 1100, endpoint=False)  # step-size: 0.01
    for rate in rateList:
        x_opt[:, 0] = rate
        prob1 = clf.predict_proba(x_opt)[:, 1]
        prob.append(prob1)
        amount = x_opt[:, 2]
        term = x_opt[:, 4]
        profitList.append(profit(prob1, amount, rate, term))
    index, value = max(enumerate(profitList), key=operator.itemgetter(1))
    return rateList[index], value, prob, rateList, profitList


def twoClusters(prob):
    """
    Using Coefficient of Variation to measure the price sensitivity of each customer.
    Using K-Means algorithm to group customers into two clusters.
    """
    CV = (np.std(np.array(prob).T, axis=1) / np.mean(np.array(prob).T, axis=1)).reshape(100, 1)
    kmeans_model = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(CV)
    labels = kmeans_model.labels_
    sensitive_group = [i for i in range(len(labels)) if labels[i] == 0]
    nonsensitive_group = [i for i in range(len(labels)) if labels[i] == 1]
    x_opt_s = x_opt[sensitive_group, :]
    x_opt_n = x_opt[nonsensitive_group, :]
    return x_opt_s, x_opt_n, sensitive_group, nonsensitive_group, CV


if __name__ == "__main__":
    # Read the csv file, and convert it into the structured dict.
    print datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', required=True)
    parser.add_argument('-f2', required=True)
    args = parser.parse_args()
    trainData = readCSV(args.f1)
    optData = readCSV(args.f2)
    print "finish loading data...\n"

    x_train = trainData['data']
    y_train = trainData['target']
    x_opt = optData['data']
    y_opt = optData['target']

    print datetime.datetime.now()
    print "start building model...\n"
    clf = LogisticRegression(penalty='l1', random_state=0, class_weight='auto')
    clf = clf.fit(x_train, y_train)
    print datetime.datetime.now()
    print "finish building model...\n"

    ############# Find a single rate that maximizes the bank's expected profit.
    rate, value, prob, rateList, profitList = rateFinder(x_opt, clf)
    print "The single rate that maximizes the bank's expected profit is: %f" % rate
    print "The expected profit is: %f" % value, "\n"
    print datetime.datetime.now()
    print "part 2 question 1 is complete...\n"


    ############# Segment the customers into two groups based on their price sensitivity and compute the
    #############     expected profit maximizing rate for each segment.
    x_opt_s, x_opt_n, sensitive_group, nonsensitive_group, CV = twoClusters(prob)
    rate_s, value_s, prob_s, rateList_s, profitList_s = rateFinder(x_opt_s, clf)
    rate_n, value_n, prob_n, rateList_n, profitList_n = rateFinder(x_opt_n, clf)
    print "Number of customer in sensitive group: ", len(sensitive_group)
    print "Number of customer in non-sensitive group: ", len(nonsensitive_group), "\n"
    print "The optimal rate of price sensitive group is: %f" % rate_s
    print "The expected profit of price sensitive group is: %f" % value_s
    print "The optimal rate of price non-sensitive group is: %f" % rate_n
    print "The expected profit of price non-sensitive group is: %f" % value_n, "\n"
    print datetime.datetime.now()
    print "part 2 question 2 is complete...starting visualization...\n"


    ############# Data Visualization
    # Visualize Bank's Expected Profit as a Function of the Interest Rate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rateList, profitList)
    plt.xlabel("Interest Rate (%)")
    plt.ylabel("Bank's Expected Profit")
    plt.title("Bank's Expected Profit as a Function of the Interest Rate")
    plt.savefig('expected_profit.png')

    # Visualize Bank's Expected Profit as a Function of the Interest Rate in Price Sensitive Group
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rateList_s, profitList_s)
    plt.xlabel("Interest Rate (%)")
    plt.ylabel("Bank's Expected Profit")
    plt.title("Bank's Expected Profit as a Function of the Interest Rate \n (Sensitive)")
    plt.savefig('expected_profit_s.png')

    # Visualize Bank's Expected Profit as a Function of the Interest Rate in Price Non-Sensitive Group
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rateList_n, profitList_n)
    plt.xlabel("Interest Rate (%)")
    plt.ylabel("Bank's Expected Profit")
    plt.title("Bank's Expected Profit as a Function of the Interest Rate \n (Non-Sensitive)")
    plt.savefig('expected_profit_n.png')

    # Visualize two groups of customers
    x = np.linspace(1, 100, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, CV)
    plt.xlabel("Observation")
    plt.ylabel("Price Sensitivity")
    plt.text(0.15, 0.8, "Sensitive", fontsize=15, horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes)
    plt.text(0.15, 0.3, "Non-Sensitive", fontsize=15, horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes)
    threshold = (np.min(CV[sensitive_group])+np.max(CV[nonsensitive_group]))/2
    plt.axhline(y=threshold)
    plt.savefig('two_groups_segments.png')

    print datetime.datetime.now()
    print "part 2 is complete...\n"





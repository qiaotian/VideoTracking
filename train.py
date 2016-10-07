""" Training and Testing
Author : QiaoTian
Date   : 6th Oct 2016
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

def main():
    xpath = '/Users/qiaotian/Downloads/dataset/sample1/resp_target.txt'
    ypath = '/Users/qiaotian/Downloads/dataset/sample1/label.txt'
    y = pd.read_csv(ypath, sep=',', header=None)
    X = pd.read_csv(xpath, sep=',', header=None).iloc[0:len(y),:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    #rf_params['n_estimators'] = 50
    gbdt_params = ['n_estimators': 100, 'max_depth': 3,
                   'subsample': 0.9, 'learning_rate': 0.1,
                   'min_samples_leaf': 1, 'random_state': 1234]
    #rf = RandomForestClassifier(**rf_params)
    gbdt = GradientBoostingClassifier(**gbdt_params)
    gbdt.fit(X_train, y_train)
    train_acc = gbdt.score(X_train, y_train)
    test_acc = gbdt.score(X_test, y_test)
    print train_acc, test_acc

if __name__ == '__main__':
    main()

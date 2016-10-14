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

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split

def main():
    xpath = '/Users/qiaotian/Downloads/dataset/sample1/feature.txt'
    ypath = '/Users/qiaotian/Downloads/dataset/sample1/label.txt'
    y = pd.read_csv(ypath, sep=',', header=None).iloc[:,1]
    X = pd.read_csv(xpath, sep=',', header=None).iloc[0:len(y),:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=9)
    # 1. Linear Regressor
    lr_params = {}
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_acc = lr.score(X_train, y_train)
    test_acc = lr.score(X_test, y_test)
    y_pred = lr.predict(X_test)
    print('-> Done Linear Regression: ', train_acc, test_acc, len([elem for elem in y_pred-y_test if abs(elem)<1.0])/len(y_test))

    # 2. Random Foreset Regressor
    rf_params = {'n_estimators':100}
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    y_pred = rf.predict(X_test)
    print('-> Done Random Forest Regression: ', train_acc, test_acc, len([elem for elem in y_pred-y_test if abs(elem)<1.0])/len(y_test))

    # 3. Gradient Booting Regressor
    gbdt_params = {'loss':'ls', 'n_estimators':100, 'max_depth':3,\
                   'subsample':0.9, 'learning_rate':0.1,\
                   'min_samples_leaf':1, 'random_state':1234}
    gbdt = GradientBoostingRegressor(**gbdt_params)
    gbdt.fit(X_train, y_train)
    train_acc = gbdt.score(X_train, y_train)
    test_acc = gbdt.score(X_test, y_test)
    y_pred = gbdt.predict(X_test)

    print('-> Done Gradient Boosting Regression: ', train_acc, test_acc, len([elem for elem in y_pred-y_test if abs(elem)<1.0])/len(y_test))

if __name__ == '__main__':
    main()

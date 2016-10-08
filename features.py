""" Feature Selection
Author : QiaoTian
Date   : 6th Oct 2016
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mutual_info_score


def main():
    xpath = '/Users/qiaotian/Downloads/dataset/sample1/feature.txt'
    ypath = '/Users/qiaotian/Downloads/dataset/sample1/label.txt'
    y = pd.read_csv(ypath, sep=',', header=None).iloc[:,1].to_frame() # to_frame is used to convert series to dataframe
    X = pd.read_csv(xpath, sep=',', header=None).iloc[0:len(y),:]

    MI = np.zeros(len(X.iloc[0,:])) # mutual information
    for i in range(len(MI)):
        #print(X.iloc[:,i].as_matrix())
        #print(y.as_matrix().ravel())
        MI[i] = mutual_info_score(X.iloc[:, i].as_matrix(), y.as_matrix().ravel())

    plt.figure(1)
    plt.plot(MI)
    plt.xlabel("feature_index")
    plt.ylabel("MI")

    mi_rank = np.argsort(np.abs(MI))[-1::-1]
    plt.figure(2)
    plt.plot(MI[mi_rank])
    plt.ylabel("MI")
    plt.show()

if __name__ == '__main__':
    main()

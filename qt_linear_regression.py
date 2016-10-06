""" Linear Regression
Author : QiaoTian
Date   : 16th Sep 2016
Revised: 16th Sep 2014
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
The origin data format:
    | date | time(s) | x1 | y1 | z1 | a1 | b1 | c1 | d1 |
    | date | time(s) | x2 | y2 | z2 | a2 | b2 | c2 | d2 |
    ......
    | date | time(s) | xn | yn | zn | an | bn | cn | dn |

The target data format:
    | time(ms) | x1 | y1 | z1 | a1 | b1 | c1 | d1 | ... | x8 | y8 | z8 | a8 | b8 | c8 | d8 |
    | time(ms) | x9 | y9 | z9 | a9 | b9 | c9 | d9 | ........................................
    | time(ms) ........................................ | xk | yk | zk | ak | bk | ck | dk |
    where k = (n/8*8)
"""

def convertDataFormat(ipath, opath):
    # 1. convert dataframe to numpy
    df = pd.read_csv(ipath, sep=' ', header=None).as_matrix()

    # 2. data processing
    # record the start and end time
    # and then calculate the time period (unit is Second)
    cnt = int(len(df)/8) # the number of valid sample
    time_bgn = df[0,1]
    time_end = df[cnt*8-1,1]
    period = (int(time_end[0:2])*3600+int(time_end[3:5])*60+int(time_end[6:8])) -\
             (int(time_bgn[0:2])*3600+int(time_bgn[3:5])*60+int(time_bgn[6:8]))

    X = np.zeros((cnt, 57), dtype=np.float)
    for i in range(cnt):
        line = np.zeros(57, dtype=np.float)
        time = (float(period)/cnt)*i #
        line[0] = time
        for j in range(8):
            line[j*7+1:(j+1)*7+1] = df[i*8+j][2::]
        X[i] = line;

    # 3. convert numpy back to dataframe
    X = pd.DataFrame(X)
    X.to_csv(opath, header=False, index=False) # do not write the column and index names
    print('convert data format is done')

def imagesProcessing(ipath, opath):
    # ipath is directory of image sequences
    import os
    list_dirs = os.walk(ipath)
    for root, dirs, files in list_dirs:
        cnt = int(len(files)) # the size of all label
        y = np.zeros((cnt, 2), dtype=np.integer) # the 1st column is horizon offset and the 2nd is vertical offset
        for f in files:
            # image processing here
            print(f)
        df = pd.DataFrame(y)
        df.to_csv(opath, header=False, index=False)
        print('image processing is done')

def drawShift(ipath):
    y = pd.read_csv(ipath, sep=' ', header=None).as_matrix() # y has two columns
    x = np.arange(len(y))
    plt.plot(x, y[:,0], color='green', label='horizon')
    plt.plot(x, y[:,1], color='red',   label='vertical')
    plt.xlabel('time(s)')
    plt.ylabel('shift(pixel)')
    plt.show()


def main():
    path = '/Users/qiaotian/Downloads/dataset/'
    
    # 1. convert the data from origin format to desired format
    convertDataFormat(path+'resp_origin.txt', path+'resp_target.txt')

    # 2. image processing and store the label in label file
    imagesProcessing(path+'images/', path+'label.txt')
    #drawShift(label)

    # 3. plot data for basis analysis
    #X = pd.read_csv(opath)



if __name__ == '__main__':
    main()

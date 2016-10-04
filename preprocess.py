""" Data Preprocess
Author : QiaoTian
Date   : 6th Oct 2016
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
The training data format:
    | date | time(s) | x1 | y1 | z1 | a1 | b1 | c1 | d1 |
    | date | time(s) | x2 | y2 | z2 | a2 | b2 | c2 | d2 |
    ......
    | date | time(s) | xn | yn | zn | an | bn | cn | dn |

The training data format revised:
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


"""
The label data format:
    | time(s) | horizontal offset(mm) | vertical offset(mm) |
"""
def imagesProcessing(ipath, opath, fps):
    # ipath is directory of image sequences
    import os
    list_dirs = os.walk(ipath)

    """ find the position edge(white) center """
    def center(image, threshold, roi, scale):
        # image: source image
        # threshold: for brightness
        # roi:
        # scale: the scale for source image, the distance divides number of pixels
        assert image.ndim == 3 # 3 channel image
        sum_row = 0
        sum_col = 0
        count = 0
        heart = [0, 0]
        for i in range(roi[1], roi[1]+roi[3]):
            for j in range(roi[0], roi[0]+roi[2]):
                if image[i][j][0] > threshold or image[i][j][1] > threshold or image[i][j][2] > threshold:
                    image[i][j] = [255,255,255] # white
                    sum_row = sum_row + i
                    sum_col = sum_col + j
                    count = count+1
                else:
                    image[i][j] = 0 # black
        heart[0] = sum_row//count*scale # unit is mm
        heart[1] = sum_col//count*scale # unit is mm
        return heart

    for root, dirs, files in list_dirs:
        cnt = int(len(files)) # the size of all label
        y = np.zeros((cnt, 3), dtype=np.float) # the 1st column is horizon offset and the 2nd is vertical offset
        y[:,0] = np.linspace(0.0, 1.0/fps*cnt, cnt, endpoint=False) # setup the 'time' column

        for i in range(len(files)):
            # image processing algorithm here
            image_dir = ipath+files[i]
            from scipy import misc
            image = misc.imread(image_dir)
            w = image.shape[0]
            h = image.shape[1]
            threshold = 150
            # roi for sample1 [581, 122, 140, 190]
            # roi for sample2 [421, 217, 130, 200]
            roi = [581, 122, 140, 190] # start w, start h, width, height
            heart = center(image, threshold, roi, 250.0/941.0)
            y[i,1:3] = heart

        df = pd.DataFrame(y)
        df.to_csv(opath, header=False, index=False)


def drawShift(ipath):
    y = pd.read_csv(ipath, sep=',', header=None).as_matrix() # y has three columns
    x = y[:,0]
    plt.plot(x, y[:,1], color='green', label='horizon')
    plt.plot(x, y[:,2], color='red',   label='vertical')
    plt.xlabel('time (/s)')
    plt.ylabel('shift (/mm)')
    plt.show()


def main():
    # Parameters
    path = '/Users/qiaotian/Downloads/dataset/sample2/'
    fps = 19 # image frequence per second, sample 1 is 18, sample 2 is 19

    # 1. convert the data from origin format to desired format
    convertDataFormat(path+'resp_origin.txt', path+'resp_target.txt')
    print('-> Done converting data format')

    # 2. image processing and store the label in label file
    imagesProcessing(path+'images/', path+'label.txt', fps)
    print('-> Done processing images')

    # 3. plot the shift trend
    drawShift(path+'label.txt')
    print('-> Done drawing shift trend in two dimentions')




if __name__ == '__main__':
    main()

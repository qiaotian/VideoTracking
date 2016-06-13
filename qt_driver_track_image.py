# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-13T18:22:30+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:29:38+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import qt_feature

from sklearn.cluster import KMeans
from six.moves import urllib
from scipy.stats import entropy


def main(argv=None):

    parser = argparse.ArgumentParser(
        description='Extract and return features from input image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #parser.add_argument('--image', default='../data/test1.jpg')
    parser.add_argument('--image', default='../../Desktop/test.png')
    args = parser.parse_args()

    image = cv2.imread(args.image)/255.0

    # set roi of the input
    roi = image[20:, 50:, :]

    # expand the dim
    if roi.ndim == 3:
        roi = roi[np.newaxis, ...] # (1,172,200,3)

    # extract features
    ftrs = features(roi)

    # kmeans to cluster
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(np.reshape(ftrs, (ftrs.shape[0]*ftrs.shape[1]*ftrs.shape[2], ftrs.shape[3])))

    # show the image
    disp = roi.reshape((roi.shape[0]*roi.shape[1]*roi.shape[2], roi.shape[3]))

    for i in xrange(len(kmeans.labels_)):
        if kmeans.labels_[i] == 0:
            disp[i] = [255,0,0]
        elif kmeans.labels_[i] == 1:
            disp[i] = [0,255,0]
        elif kmeans.labels_[i] == 2:
            disp[i] = [0,0,255]
        elif kmeans.labels_[i] == 3:
            disp[i] = [255,0,255]
        elif kmeans.labels_[i] == 4:
            disp[i] = [0,255,255]
        elif kmeans.labels_[i] == 5:
            disp[i] = [255,255,255]
        else:
            disp[i] = [255, 255, 0]

    disp.resize(roi.shape[1], roi.shape[2], 3)
    print(disp.shape)
    cv2.imwrite("./culsters_%s_out.jpg" % args.image[-8:-4], disp)

if __name__ == '__main__':
    main()

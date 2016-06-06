#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @Author: Tian Qiao
# @Date:   2016-06-01T14:54:45+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   root
# @Last modified time: 2016-06-05T14:00:49+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE


import tensorflow as tf
import numpy as np
import argparse
import cv2
#import gzip
#import os
#import re
#import sys
#import tarfile

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import KMeans
from six.moves import urllib
from scipy.stats import entropy

# Global constants describing the US data

def features(image):
    sess = tf.Session()

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    """ mean """
    mean = tf.to_float(tf.nn.avg_pool(image,\
                          ksize=[1,5,5,1],\
                          strides=[1,1,1,1],\
                          padding='SAME',\
                          data_format='NHWC', name=None), name='ToFloat')
    print('Dim of feature1 is:', mean.get_shape())

    """ standard deviation """
    stddev = tf.nn.conv2d(tf.to_float((image - mean)**2//(25-1), name='ToFloat'),\
                          tf.ones([5,5,3,3],\
                          dtype=tf.float32),\
                          strides=[1,1,1,1],\
                          padding='SAME')
    print('Dim of feature2 is:', stddev.get_shape())

    """ entropy """
    """
    entr = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if(i>1 and i<height-2 and j>1 and j<width-2):
                temp = tf.reshape(image[0, i-2:i+3, j-2:j+3, 0], [25,1]).eval(session=tf.Session()) # (25,1)
                entr[i,j] = entropy(temp)

    ftrs = np.concatenate((mean, stddev, entr), axis=0)
    """

    ftrs = tf.concat(3, [mean, stddev])
    print('Dim of features is:', ftrs.get_shape())

    # convert tensor to numpy array
    ftrs = sess.run(ftrs)
    return ftrs


def main(argv=None):
    sess = tf.Session()

    parser = argparse.ArgumentParser(
        description='Extract and return features from input image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--image')
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
    kmeans = KMeans(n_clusters=4)
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
        """
        if kmeans.labels_[i] == 1:
            disp[i] = [0, 255, 0]
        else:
            disp[i] = [0, 0, 0]
        """
    disp.resize(roi.shape[1], roi.shape[2], 3)
    print(disp.shape)
    cv2.imwrite("./res/out.jpg", disp)


if __name__ == '__main__':
    main()


    # fearures extract
    # filenames = [args.image]
    # Create a queue that produces the filenames to read
    # filename_queue = tf.train.string_input_producer(filenames)
    # reader = tf.WholeFileReader()
    # key, value = reader.read(filename_queue)
    # img = tf.image.decode_jpeg(value, channels=3)

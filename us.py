# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib

import gzip
import os
import re
import sys
import tarfile
import scipy
import tensorflow as tf
import argparse

import us_input

# Global constants describing the US data

def inference(image):
    # mean
    mean = tf.nn.avg_pool(image, ksize=[5,5,1],\
                          strides=[1,1,1], padding='SAME',\
                          data_format='NHWC', name='mean')

    # standard deviation
    aver = (image - mean)**2//(25-1)
    kernel = tf.get_variable('kernel', shape=[5,5,1],\
                          initializer=tf.constant_initializer(value=1.0, dtype=tf.float))
    stddev = tf.nn.conv2d(aver, kernel, strides=[1,1,1], padding='SAME', use_cudnn_on_gpu=None, data_format=None, name='stddev')

    # entropy
    entropy = np.zeros((image.shape.height, image.shape.width))
    for i in image.shape.height:
        for j in image.shape.width:
            if(i>1 && i<image.shape.height-2 && j>1 && j<image.shape.width-2):
                [i,j] = scipy.stats.entropy(image[i-2:i+2, j-2:j+2].reshape((5*5,1)))
    ftrs = np.concatenate(mean, stddev, entropy)

def main(argv=None):
    parser = argparse.ArgumentParser(
        descrption='Extract and return features from input image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

if __name__ == '__main__':
    main()

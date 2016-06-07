#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @Author: Tian Qiao <root>
# @Date:   2016-06-05T14:01:18+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   root
# @Last modified time: 2016-06-07T15:55:36+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import amin
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def template(image):
    return

def corre_coefficient(source, template):
    print('souce shape is:', source.shape)
    print('template shape is:', template.shape)
    # template: template image
    # source: the image should be compared with template
    src_height = source.shape[0]
    src_width = source.shape[1]

    temp_height = template.shape[0]
    temp_width = template.shape[1]

    ans = np.zeros((src_height, src_width))

    if src_height < temp_height or src_width < temp_width:
        print('The template image is larger than input image')
        return None

    # 1. calculate the normalized template image
    aver_template = np.sum(template) // (temp_height*temp_width)
    norm_template = template - np.ones((temp_height, temp_width), dtype=np.float32) * aver_template

    # 2. calculate the normalized source image
    norm_source = np.zeros((temp_height, temp_width))
    for i in range(src_height):
        for j in range(src_width):
            # calculate correlated coefficient for every pixel in target image
            norm_source[:min(temp_height, src_height-i), :min(temp_width, src_width-j)] =\
                source[i:min(i+temp_height, src_height), j:min(j+temp_width, src_width)]

            aver_source = np.sum(norm_source) // (temp_height*temp_width)
            norm_source = norm_source - np.ones((temp_height, temp_width), dtype=np.float32) * aver_source
            cv2.imshow('normal source image', norm_source)

            correlated_coefficient = np.sum(np.multiply(norm_source, norm_template))//\
                                     (np.sum(np.square(norm_source))+1)*\
                                     (np.sum(np.square(norm_template))+1)
            ans[i,j] = correlated_coefficient
    return ans

def main():
    parser = argparse.ArgumentParser(
        description='Extract and return features from input image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', help='source image', default='./res/test1.jpg')
    parser.add_argument('--tmp', help='template image', default='./res/template40x20.jpg')
    args = parser.parse_args()

    src = cv2.cvtColor(cv2.imread(args.src), cv2.COLOR_RGB2GRAY) # single channel image
    tmp = cv2.cvtColor(cv2.imread(args.tmp), cv2.COLOR_RGB2GRAY) # single channel image

    #cv2.imshow('source image', src)
    #cv2.imshow('template image', tmp)

    #cc = corre_coefficient(src, tmp)
    #print(cc.shape)

    # plot the data
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0,3)
    Y = np.arange(0,3)
    X, Y = np.meshgrid(X,Y)
    Z = X+Y
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = list(range(1,11)) for x in range(src.shape[0])
    Y = list(range(1,11)) for x in range(src.shape[1])
    Z = np.ones(src.shape[0], src.shape[1])
    plt.show()
    """
if __name__ == '__main__':
    main()

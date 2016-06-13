#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @Author: Tian Qiao <root>
# @Date:   2016-06-05T14:01:18+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:24:09+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE


import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import amin
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


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
    #aver_template = np.sum(template) // (temp_height*temp_width)
    #norm_template = template - np.ones((temp_height, temp_width), dtype=np.float32) * aver_template
    norm_template = np.float32(template)

    # 2. calculate the normalized source image
    norm_source = np.float32(np.zeros((temp_height, temp_width)))
    for i in range(src_height):
        for j in range(src_width):
            # calculate correlated coefficient for every pixel in target image
            norm_source[:min(temp_height, src_height-i), :min(temp_width, src_width-j)] =\
                source[i:min(i+temp_height, src_height), j:min(j+temp_width, src_width)]

            # aver_source = np.sum(norm_source) // (temp_height*temp_width)
            # norm_source = norm_source - np.ones((temp_height, temp_width), dtype=np.float32) * aver_source

            len_src   = np.float32(np.sqrt(np.sum(np.square(norm_source))))
            len_tmp   = np.float32(np.sqrt(np.sum(np.square(norm_template))))
            sum_cross = np.float32(np.sum(np.multiply(norm_source, norm_template)))

            correlated_coefficient = (sum_cross+1)/(len_src+1)/(len_tmp+1)

            ans[i,j] = correlated_coefficient
    return ans

def main():
    parser = argparse.ArgumentParser(
        description='Extract and return features from input image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', help='source image', default='../ExperimentData/test1.jpg')
    parser.add_argument('--tmp', help='template image', default='../ExperimentData/template114x167.jpg')
    args = parser.parse_args()

    src = cv2.cvtColor(cv2.imread(args.src), cv2.COLOR_RGB2GRAY) # single channel image
    tmp = cv2.cvtColor(cv2.imread(args.tmp), cv2.COLOR_RGB2GRAY) # single channel image

    cc = corre_coefficient(src, tmp)

    """ 3D colorbar plot """
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, cc.shape[1], 1)
    Y = np.arange(0, cc.shape[0], 1)
    X, Y = np.meshgrid(X,Y)
    Z = cc
    #X = np.arange(-5, 5, 0.25)
    #Y = np.arange(-5, 5, 0.25)
    #X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)

    surf = ax.plot_surface(X, Y, Z, rstride=15, cstride=15, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=1, aspect=20) # set color bar location and size
    plt.show()
    """

    """ 2D colorbar plot """
    fig = plt.gcf()

    ax = fig.add_subplot(111)
    X = np.arange(0, cc.shape[1], 1)
    Y = np.arange(0, cc.shape[0], 1)
    X,Y = np.meshgrid(X,Y)
    quadmesh = ax.pcolormesh(X,Y,cc)
    quadmesh.set_clim(vmin=-0.2, vmax=1.5)
    plt.show()

    """ wireframe plot """
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    X = np.arange(0,cc.shape[1], 1)
    Y = np.arange(0,cc.shape[0], 1)
    X, Y = np.meshgrid(X,Y)
    Z = cc

    print(X.shape)
    print(Y.shape)
    print(Z.shape)

    ax.plot_surface(X, Y, Z, rstride=15, cstride=15, alpha=0.3)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    #ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    #ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    #ax.set_zlim(-100, 100)

    plt.show()
    """

if __name__ == '__main__':
    main()

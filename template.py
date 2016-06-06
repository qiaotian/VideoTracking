#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @Author: Tian Qiao <root>
# @Date:   2016-06-05T14:01:18+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   root
# @Last modified time: 2016-06-05T16:10:45+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from numpy import amin

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

    for i in range(src_height):
        for j in range(src_width):
            # calculate correlated coefficient for every pixel in target image
            src_mean = 0
            temp_mean = 0
            for m in range(temp_height):
                for n in range(temp_width):
                    if i+m>=src_height or j+n>=src_width:
                        src_mean = src_mean+0
                        temp_mean = temp_mean+template[m,n]
                        continue
                    src_mean = src_mean+source[m,n]
                    temp_mean = temp_mean+template[m,n]
            src_mean = src_mean/(temp_height*temp_width)
            temp_mean = temp_mean/(temp_height*temp_width)

            # construct a new array that pads zeros to the matrix if index beyond boundry
            padding = np.zeros((temp_height, temp_width))
            #x = source[i:amin([temp_height, src_height-i]), j:amin([temp_width, src_width-i])]
            #y = padding[i:amin([temp_height, src_height-j]), j:amin([temp_width, src_width-j])]
            #print(x.shape)
            #print(y.shape)

            padding[:min(temp_height, src_height-i), :min(temp_width, src_width-j)] = source[i:min(i+temp_height, src_height), j:min(j+temp_width, src_width)]

            norm_src = padding - np.ones((temp_height, temp_width))*src_mean
            norm_temp = template-np.ones((temp_height, temp_width))*temp_mean
            factor0 = 0
            factor1 = 0
            factor2 = 0
            for m in range(temp_height):
                for n in range(temp_width):
                    factor0 = factor0 + norm_src[m,n]*norm_temp[m,n]
                    factor1 = factor1 + norm_src[m,n]*norm_src[m,n]
                    factor2 = factor2 + norm_temp[m,n]*norm_temp[m,n]
            correlated_coefficient = factor0//(factor1+1)//(factor2+1)
            ans[i,j] = correlated_coefficient
    return ans

def main():
    parser = argparse.ArgumentParser(
        description='Extract and return features from input image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', help='source image')
    parser.add_argument('--tmp', help='template image')
    args = parser.parse_args()

    src = cv2.cvtColor(cv2.imread(args.src), cv2.COLOR_RGB2GRAY) # single channel image
    tmp = cv2.cvtColor(cv2.imread(args.tmp), cv2.COLOR_RGB2GRAY) # single channel image

    cc = corre_coefficient(src, tmp)
    print(cc.shape)


if __name__ == '__main__':
    main()

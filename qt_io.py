# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-14T10:49:58+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-14T13:16:18+08:00
# @License: Free License

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_txt(input_filename):
    array = np.loadtxt(input_filename)
    start = np.tile(array[0], (len(array),1))
    dist = np.sqrt(np.sum((array-start)**2, axis=1))
    return dist


def read_txt_with_timestamp(input_filename):
    return

def draw_scatter(array):
    x = np.linspace(0, 1, len(array))
    plt.figure()
    plt.plot(x, array)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='draw scatter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_filename', default='./res/hightlight_center_log.txt', help='')
    args = parser.parse_args()

    draw_scatter(read_txt(args.input_filename))

    return

if __name__ == '__main__':
    main()

# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-13T18:25:45+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:28:37+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE

import cv2
import numpy as np
import argparse
import time
import os
import qt_utility

def main(arv=None):
    parser = argparse.ArgumentParser(
        description='Crop the image and store the croped image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--image')
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--w', type=int, help='width')
    parser.add_argument('--h', type=int, help='height')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    croped = roi(image, args.x, args.y, args.w, args.h)
    if croped.shape[0]!=0:
        cv2.imwrite('./out/%s.jpg' % time.time(), croped)
    else:
        print('croped image is None')
    """
    """
    parser.add_argument('-i', '--input_filename', default='../ExperimentData/usdata/MOVIE-0001.mp4')
    parser.add_argument('-o', '--output_filename', default='../ExperimentData/usdata/croped_MOVIE-0001.avi')
    parser.add_argument('-x', '--origin_x', type = int, default=300)
    parser.add_argument('-y', '--origin_y', type = int, default=300)
    parser.add_argument('-w', '--width', type = int, default=50)
    parser.add_argument('-h', '--height', type = int, default=50)

    args = parser.parse_args()
    qt_utility.crop_video(args.input_dir, args.output_dir, args.x, args.y, args.w, args.h)

if __name__ == '__main__':
    main()

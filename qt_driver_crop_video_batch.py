# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-13T18:25:45+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:38:26+08:00
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

    parser.add_argument('--input_dir', default='../ExperimentData/usdata/', help='')
    parser.add_argument('--output_dir', default='../ExperimentData/usdata/', help='')
    parser.add_argument('--origin_x', type = int, default=30, help='')
    parser.add_argument('--origin_y', type = int, default=160, help='')
    parser.add_argument('--width', type = int, default=480, help='')
    parser.add_argument('--height', type = int, default=200, help='')

    args = parser.parse_args()
    qt_utility.crop_video_batch(args.input_dir, args.output_dir, args.origin_x, args.origin_y, args.width, args.height)

if __name__ == '__main__':
    main()

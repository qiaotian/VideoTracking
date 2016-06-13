# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-13T18:21:07+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:29:18+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from sklearn.cluster import KMeans
from six.moves import urllib
from scipy.stats import entropy
import qt_feature

def main(argv=None):

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_filename', default='../ExperimentData/usdata/croped_MOVIE-0012.mp4', help='')
    parser.add_argument('--output_filename', default='../ExperimentData/clustered_MOVIE-0012.mp4', help='')

    args = parser.parse_args()
    videoCapture = cv2.VideoCapture(args.input_filename)

    fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                   int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')
    videoWriter  = cv2.VideoWriter(args.output_filename, fourcc, fps, size, True)

    while(videoCapture.isOpened()):
        ret, frame = videoCapture.read()
        if ret==True:
            image = frame[np.newaxis, ...]
            ftrs = features(image)
            kmeans = KMeans(n_clusters = 3)
            kmeans.fit(np.reshape(ftrs, (ftrs.shape[0]*ftrs.shape[1]*ftrs.shape[2], ftrs.shape[3])))
            disp = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], image.shape[3]))
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
            disp.resize(image.shape[1], image.shape[2], 3)
            #cv2.imwrite(args.output_filename, disp)
            videoWriter.write(disp)

if __name__ == '__main__':
    main()

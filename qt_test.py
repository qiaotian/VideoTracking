# @Author: Tian Qiao <root>
# @Date:   2016-05-27T07:31:13+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:31:37+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE



 # -*- coding: utf-8 -*-
# $File: io.py
# $Date: Fri May 27 07:56:30 2016 +0800
# $Author: qiaotian <qiaotian@me.com>

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from feature import features

from sklearn.cluster import KMeans
from six.moves import urllib

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import time
# import utils

"""
# args parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())

# convert from BGR to RGB, which could be displayed by matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
height = image.shape[0]
width = image.shape[1]
image = image.reshape((height * width, 3))

# cluster the pixels intensities
cluster = KMeans(n_clusters = args["clusters"])
cluster.fit(image)
# print cluster.labels_[1:1000]

image_copy = image
for i in xrange(len(cluster.labels_)):
    if cluster.labels_[i] == 0:
        image_copy[i] = [255,0,0]
    elif cluster.labels_[i] == 1:
        image_copy[i] = [0,255,0]
    elif cluster.labels_[i] == 2:
        image_copy[i] = [0,0,255]
    elif cluster.labels_[i] == 3:
        image_copy[i] = [255,0,255]
    elif cluster.labels_[i] == 4:
        image_copy[i] = [0,255,255]
    elif cluster.labels_[i] == 5:
        image_copy[i] = [255,255,255]
    else:
        image_copy[i] = [255, 255, 0]
image_copy = image_copy.reshape((height, width, 3))
cv2.imwrite("out.jpg", image_copy)
"""

def pooling(image):
    height, width = image.shape
    image_copy = image.copy()
    for i in range(height):
        for j in range(width):
            if i-1 < 0 or j-1 <0 or i+1>height-1 or j+1>width-1:
                image_copy[i, j] = image[i,j]
            else:
                for m in range(3):
                    for n in range(3):
                        image_copy[i, j] = min(image_copy[i, j], image[i-m, j-n])
    return image_copy

#logger = logging.getLogger(__name__)

input_filename = '../../Downloads/usliverseq/volunteer04.avi'
output_filename = '../ExperimentData/' + time.strftime("%m%d-%H-%M-%S") + '.avi'

videoCapture = cv2.VideoCapture(input_filename)

fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

# fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G') # doesn't work
# fourcc = cv2.cv.CV_FOURCC('I', '4', '2', '0') # doesn't work
fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S') # works
videoWriter = cv2.VideoWriter(output_filename, fourcc, fps, size, True)

while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()
    print(frame.shape)
    if ret==True:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # expand the dim
        if image.ndim == 3:
            image = image[np.newaxis, ...] # (1,172,200,3)
        # extract features
        ftrs = features(image)
        # kmeans to cluster
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(np.reshape(ftrs, (ftrs.shape[0]*ftrs.shape[1]*ftrs.shape[2], ftrs.shape[3])))

        # show the image
        disp = image[0]
        disp.resize(disp.shape[0]*disp.shape[1], 3)
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
            if kmeans.labels_[i] == 2:
                disp[i] = [0, 255, 255]
            else:
                disp[i] = [0, 0, 0]
            """
        disp.resize(image.shape[1], image.shape[2], 3)
        print(disp.shape)

        videoWriter.write(disp)

        #cv2.imshow('frame', disp)
        #out.write(pool)
        # kmeans = KMeans(n_clusters=4)
        # kmeans.fit(gray.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()

 # -*- coding: utf-8 -*-
# $File: io.py
# $Date: Fri May 27 07:56:30 2016 +0800
# $Author: qiaotian <qiaotian@me.com>

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import argparse
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

#cap = cv2.VideoCapture('../../../Downloads/usliverseq/volunteer02.avi')
#cap = cv2.VideoCapture('../../../Downloads/usliverseq/volunteer03.avi')
cap = cv2.VideoCapture('../../../Downloads/usliverseq/volunteer04.avi')
#cap = cv2.VideoCapture('../../../Downloads/usliverseq/volunteer05.avi')

# Define the codec and create VideoWriter object
# height = 0
# width = 0
# if(cap.isOpened()):
#    ret, frame = cap.read()
#    height, width = frame.shape

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (height,width))

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("%d.jpg" % (count) , image)
        # count = count+1
        # pooling
        # pool = pooling(gray)

        cv2.imshow('frame', image)
        #out.write(pool)
        # kmeans = KMeans(n_clusters=4)
        # kmeans.fit(gray.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()



# Ultrasound images reader
# class SeqReader(object):
#    _fin_data = None;
#    _cur_obj = None;

#    scenes = None;

#    def __init__(self, fpath):
#        self.scenes = []
#        logger.infor('load ultrasound images data file {}'.format(fpath))

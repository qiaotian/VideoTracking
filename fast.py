# @Author: Tian Qiao <qiaotian>
# @Date:   2016-11-17T14:44:17+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-11-17T14:44:17+08:00



import cv2
import numpy as np

filename = "./res/ct_liver.png"

img = cv2.imread(filename)
gray = cv2.cvtColor(img, COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.connerHarris(gray, 2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

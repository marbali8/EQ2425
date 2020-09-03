import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

import collections
from sift import siftFeatures

def checkDistance(point1, point2):
    th = 2
    # print(np.linalg.norm(point1 - point2))
    if np.abs(point1[0] - point2[0]) <= th and np.abs(point1[1] - point2[1]) <= th:
        return distance.euclidean(point1, point2)
    return -1

def rotateImage(img, deg, show = False):
    shape1 = img.shape[0]
    shape2 = img.shape[1]
    img2 = ndimage.rotate(img, deg)
    # img2 = cv2.resize(img2, (shape1, shape2))
    if show:
        plt.imshow(img2)
        plt.show()
    return img2

img1 = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_COLOR)

img2 = cv2.imread('./data1/obj1_t1.JPG', cv2.IMREAD_COLOR)

kp1, dp1 = siftFeatures(img1, './', True)


dp1_sorted = [x.sort() for x in dp1] # sorting the histograms
x = []
y = []
for teta in range(0, 360+15, 15):

    img2 = rotateImage(img1, teta)
    kp2, dp2 = siftFeatures(img2)

    count = 0 # counting matches
    dp2_sorted = [tmp.sort() for tmp in dp2]

    for point1 in dp1:
        for point2 in dp2:
            if distance.euclidean(point1, point2) < 30:
               count += 1
               break
    y.append(count / len(kp1))
    x.append(teta)
    print(teta, count / len(kp1))

plt.plot(x,y)
plt.show()
# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)

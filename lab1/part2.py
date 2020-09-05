import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

import collections
from sift import *

def checkDistance(point1, point2):

    th = 2
    # print(np.linalg.norm(point1 - point2))
    if np.abs(point1[0] - point2[0]) <= th and np.abs(point1[1] - point2[1]) <= th:
        return distance.euclidean(point1, point2)
    return -1

def rotateImage(img, deg, show = False):

    # shape1 = img.shape[0]
    # shape2 = img.shape[1]
    img2 = ndimage.rotate(img, deg)
    # img2 = cv2.resize(img2, (shape1, shape2))
    if show:
        plt.imshow(img2)
        plt.show()
    return img2


img1 = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_COLOR)
kp1, dp1 = siftFeatures(img1)
kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])

x = []
y = []
for teta in np.arange(0, 360+15, 15):

    img2 = rotateImage(img1, teta)
    kp2, dp2 = siftFeatures(img2)

    kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])

    kp2 -= np.array([img2.shape[1]/2, img2.shape[0]/2]).reshape((1, -1))
    teta_r = teta*np.pi/180
    R = np.array([[np.cos(teta_r), -np.sin(teta_r)], [np.sin(teta_r), np.cos(teta_r)]])
    kp = R @ kp2.T + np.array([img1.shape[1]/2, img1.shape[0]/2]).T.reshape((-1, 1))

    count = 0 # counting matches
    for i, point1 in enumerate(kp.T):
        best = (-1, distance.euclidean([0, 0], [img1.shape[1], img1.shape[0]])) # (index, distance)
        for j, point2 in enumerate(kp1):

            if np.abs(point1[0] - point2[0]) <= 2 and np.abs(point1[1] - point2[1]) <= 2:
                # if best[0] != -1:
                    # print('a', end = ' ')
                if distance.euclidean(point1, point2) < best[1]:

                    count += (best[0] == -1)
                    best = (j, distance.euclidean(point1, point2))

        # print("")
        np.delete(kp1, best[0])

    y.append(count / len(kp1))
    x.append(teta)
    print(teta, count / len(kp1))

plt.plot(x,y)
plt.ylim((0, 1))
plt.show()
# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)

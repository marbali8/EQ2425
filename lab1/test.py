import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

import collections
from sift import *
from part2 import rotateImage

teta = 15
teta_r = 15*np.pi/180

img1 = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_COLOR)
kp1, dp1 = siftFeatures(img1)

img2 = rotateImage(img1, teta)
kp2, dp2 = siftFeatures(img2)

kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
kp2 -= np.array([img2.shape[1]/2, img2.shape[0]/2]).reshape((1, -1))

R = np.array([[np.cos(teta_r), -np.sin(teta_r)], [np.sin(teta_r), np.cos(teta_r)]])
kp = R @ kp2.T + np.array([img1.shape[1]/2, img1.shape[0]/2]).T.reshape((-1, 1))

plotSift(img1, kp1)
plt.scatter(kp[0, :], kp[1, :], c = 'black')
plt.show()

kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])
count = 0
for point1 in kp.T:
    for point2 in kp1:
        if np.abs(point1[0] - point2[0]) <= 2 and np.abs(point1[1] - point2[1]) <= 2:
           count += 1
           break
print(count / len(kp1))

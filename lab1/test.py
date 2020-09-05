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

teta = 5

img1 = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_COLOR)
kp1, dp1 = siftFeatures(img1, None, False)
kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])

img2 = rotateImage(img1, teta)
kp2, dp2 = siftFeatures(img2, None, True)

kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
R = np.array([[np.cos(-teta), -np.sin(-teta)], [np.sin(-teta), np.cos(-teta)]])
kp = R @ kp2.T
print(kp1[:5], kp.T[:5])
# plt.scatter(kp[:, 0], kp[:, 1])
# plt.show()

count = 0
for point1 in kp.T:
    for point2 in kp1:
        if np.abs(point1[0] - point2[0]) <= 50 and np.abs(point1[1] - point2[1]) <= 50:
           count += 1
           break
print(count / len(kp1))

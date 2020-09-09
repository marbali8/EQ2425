#FEATUREs MATCHING


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


img1 = cv2.imread('./data1/obj1.JPG', cv2.IMREAD_COLOR)

img2 = cv2.imread('./data1/obj2.JPG', cv2.IMREAD_COLOR)

kp1, dp1 = siftFeatures(img1)
kp2, dp2 = siftFeatures(img2)
'''
plotSift(img1,kp1)
plt.figure() #Create a new one
plotSift(img2,kp2)
'''
kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])
kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
[x.sort() for x in dp1 ]
[x.sort() for x in dp2 ]

#60 100%

TH = 25

count = 0
matched = []
matched_hist = []
for i, point1 enumerate(dp1):
    for coord2, point2 in enumerate(dp2):
        if distance.euclidean(point1, point2) <= TH:
            count+=1
            matched.append([coord1,coord2])
            matched_hist.append([point1,point2])
            np.delete(kp2,coord2)
            np.delete(dp2,point2)
            break
            


print(count)
matched = np.array(matched)
c = [(random.random(), random.random(), random.random()) for k in matched]
plt.figure()
newPlotSift(img1,matched[:,0],None,c)
plt.figure()
newPlotSift(img2,matched[:,1],None,c)

plt.figure()
plt.bar(range(128), matched_hist[0][0])
plt.bar(range(128), matched_hist[0][1])
plt.show()


# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)


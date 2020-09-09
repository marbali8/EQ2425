# FEATURE MATCHING

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

import collections
from sift import *

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
[x.sort() for x in dp1]
[x.sort() for x in dp2]

#60 100%

TH = 25
count = 0
matched = []
matched_hist = []
for coord1, point1 in zip(kp1, dp1):
    for coord2, point2 in zip(kp2, dp2):
        if distance.euclidean(point1, point2) <= TH:
            count += 1
            matched.append([coord1, coord2])
            matched_hist.append([point1, point2])
            np.delete(kp2, coord2)
            np.delete(dp2, point2)
            break

print('matched keypoints (threshold): ' + str(count))
matched = np.array(matched)
c = [(random.random(), random.random(), random.random()) for k in matched]
plt.figure()
newPlotSift(img1, matched[:, 0], None, c)
plt.figure()
newPlotSift(img2, matched[:, 1], None, c)

# plt.figure()
# plt.bar(range(128), matched_hist[0][0])
# plt.bar(range(128), matched_hist[0][1])
plt.show()

# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)

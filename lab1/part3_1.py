# FEATURE MATCHING

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

import collections
from sift import *

img1 = cv2.imread('./data1/obj1.JPG', cv2.IMREAD_COLOR)

img2 = cv2.imread('./data1/logo.jpg', cv2.IMREAD_COLOR)
#img2 = rescaleImage(img1,0.9, show = False)
#img2 = rotateImage(img1,90,show= False)

kp1, dp1 = siftFeatures(img1)
kp2, dp2 = siftFeatures(img2)

'''
plotSift(img1,kp1)
plt.figure() #Create a new one
plotSift(img2,kp2)
'''
keypoints1 = kp1
keypoints2 = kp2
kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])
kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
#[x.sort() for x in dp1]
#[x.sort() for x in dp2]

#60 100%

TH = 160
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
'''
c = [(0.28438322199569144, 0.015497746634554277, 0.13935684627908396),
 (0.2578917402743346, 0.7262473265513699, 0.528851973747289),
 (0.9551276291945853, 0.43549274382532277, 0.6073936820039857),
 (0.9458799728805743, 0.22740299878783243, 0.20377516439735077),
 (0.9200593684528807, 0.004572325998327131, 0.9788416914610405),
 (0.15404349164208686, 0.8224780920774485, 0.4933064255494063),
 (0.14471230311386996, 0.5319432897773334, 0.6858802568959502),
 (0.06837084269624383, 0.289026879172278, 0.7285926923850355),
 (0.2482696042350313, 0.9061011914040853, 0.8722102740847987),
 (0.1604547038424181, 0.0768482998529686, 0.5356081114810737)]
'''
plt.figure()
newPlotSift(img1, matched[:, 0], None, c,50)
plt.figure()
newPlotSift(img2, matched[:, 1], None, c,50)

# plt.figure()
# plt.bar(range(128), matched_hist[0][0])
# plt.bar(range(128), matched_hist[0][1])
plt.show()

# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)

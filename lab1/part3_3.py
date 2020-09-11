#FEATUREs MATCHING


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage
from surf import *

import collections
from sift import *
from scipy import spatial
import time
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
#img2 = rotateImage(img1,15,False)

kp1, dp1 = siftFeatures(img1)
kp2, dp2 = siftFeatures(img2)
#kp1, dp1 = surfFeatures('./data1/obj1.JPG')

#kp2, dp2 = surfFeatures('./data1/obj2.JPG')
'''
plotSift(img1,kp1)
plt.figure() #Create a new one
plotSift(img2,kp2)
'''
kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])
kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
dp1_nonSorted = np.copy(dp1)
dp2_nonSorted = dp2
#[x.sort() for x in dp1 ]
#[x.sort() for x in dp2 ]




dList1 = dp1.tolist();
tree = spatial.KDTree(dList1)


dList2 = dp2.tolist()

distance , idx = tree.query(dList2,2)

#The ith element of kp1 matched is the elemente that match with kp2[i]
kp1_matched = kp1[idx[:,0]]

ratio=distance[:,0]/distance[:,1]
idxDel = np.where(ratio>0.80)
kp1_matched=np.delete(kp1_matched,idxDel,axis=0)
kp2=np.delete(kp2,idxDel,axis=0)
print(kp2.shape[0])
c = [(random.random(), random.random(), random.random()) for k in range(kp1_matched.shape[0])]
plt.figure()
newNewPlotSift(img1,kp1_matched,None,c,20)
plt.figure()
newNewPlotSift(img2,kp2,None,c,20)

plt.show()


# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)

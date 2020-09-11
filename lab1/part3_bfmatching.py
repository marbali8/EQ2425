import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

import collections
from sift import *

# help: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

img1 = cv2.imread('./data1/obj1.JPG', cv2.IMREAD_COLOR)
img2 = cv2.imread('./data1/obj2.JPG', cv2.IMREAD_COLOR)

kp1, dp1 = siftFeatures(img1)
kp2, dp2 = siftFeatures(img2)

bf = cv2.BFMatcher(crossCheck = True) # normType = cv2.NORM_L2 by default
matches = bf.match(dp1, dp2)
matches = sorted(matches, key = lambda x:x.distance)
img1_rgb = img1[:, :, ::-1] #rgb (for plt)
img2_rgb = img2[:, :, ::-1] #rgb (for plt)
img3 = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches[:20], None, flags = 2)
plt.imshow(img3), plt.title('20 first matches (' + str(len(matches)) + ' found, ' + str(len(kp1)) + ' generated)'), plt.show()

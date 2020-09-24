import cv2
import os
import glob
import numpy as np

def siftFeatures(img, nfeatures = 300):

    ## open image
    # img = cv2.imread(imgPath) #bgr (for cv2)
    img_rgb = img[:, :, ::-1] #rgb (for plt)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## search keypoints
    sift = cv2.SIFT_create(nfeatures = nfeatures)
    kp, desc = sift.detectAndCompute(img, None)

    return kp, desc

n_obj = 50

# server
# s_path = 'Data2/server/'
# os.makedirs(s_path + 'sift', exist_ok = True)
#
# for i in np.arange(n_obj)+1:
#
#     img_paths = glob.glob(s_path + 'obj' + str(i) + '_*.JPG') # all len = 3 but 26 (2), 37 (4), 38 (2)
#
#     with open(s_path + 'sift/obj' + str(i) + '.npy', 'wb') as f:
#         dps = []
#         for path in img_paths:
#
#             img = cv2.imread(path, cv2.IMREAD_COLOR)
#             _, dp = siftFeatures(img)
#             dps.append(dp[:300])
#
#         np.save(f, np.array(dps))

# client
c_path = 'Data2/client/'
os.makedirs(c_path + 'sift', exist_ok = True)

for i in np.arange(n_obj)+1:

    img_path = glob.glob(c_path + 'obj' + str(i) + '_*.JPG')[0] # all len1, so [0]

    with open(c_path + 'sift/obj' + str(i) + '_t.npy', 'wb') as f:

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, dp = siftFeatures(img)
        dp = dp[:300].reshape((1, dp[:300].shape[0], dp[:300].shape[1]))
        np.save(f, dp)

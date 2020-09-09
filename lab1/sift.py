import os
import cv2
import random
import matplotlib.pyplot as plt

# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
# https://docs.opencv.org/3.4.9/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html#a865f4dee68820970872c46241437f9cd

def siftFeatures(img):

    ## open image
    # img = cv2.imread(imgPath) #bgr (for cv2)
    img_rgb = img[:, :, ::-1] #rgb (for plt)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)

    ## search keypoints
    sift = cv2.SIFT_create(nfeatures = 30)
    kp, desc = sift.detectAndCompute(img, None)

    return kp, desc

def plotSift(img, kp, save = None):

    img_rgb = img[:, :, ::-1] #rgb (for plt)

    ## draw keypoints

    # make sure there are no kp outside the image
    # for i, k in enumerate(kp):
    #     if k.pt[0] > img.shape[1] or k.pt[1] > img.shape[0]:
    #         print(i, k.pt)

    # aux_img = img.copy()
    # out = cv2.drawKeypoints(img, kp, aux_img)
    # cv2.imwrite('./kp.jpg', out)

    # plt.figure()
    fig = plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    c = [(random.random(), random.random(), random.random()) for k in kp]
    if isinstance(kp[0], cv2.KeyPoint):
        plt.scatter([k.pt[0] for k in kp], [k.pt[1] for k in kp], s = 9, color = c)
    else:
        plt.scatter(kp[:, 0], kp[:, 1], s = 9, color = c)
    # plt.gcf().set_size_inches(img.shape[1], img.shape[0])

    if isinstance(save, str):
        plt.savefig(save) # , dpi = 1
        print('sift saved to' + save)


def newPlotSift(img, kp, save = None, c='black'):

    img_rgb = img[:, :, ::-1] #rgb (for plt)


    fig = plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout(pad = 0)

    plt.scatter([k[0] for k in kp], [k[1] for k in kp], s = 9, color = c)
    # plt.gcf().set_size_inches(img.shape[1], img.shape[0])

    if isinstance(save, str):
        plt.savefig(save) # , dpi = 1
        print('sift saved to' + save)

#Resistent to overlapping
def newNewPlotSift(img, kp, save = None, c='black'):

    img_rgb = img[:, :, ::-1] #rgb (for plt)


    fig = plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    s= [10]*kp.shape[0]
    for i, point in enumerate(kp):
        if i==0:
            continue
        else:
            if (np.all(np.isin(kp[i],kp[0:i]))):
                s[np.where(np.all(kp[i] == kp[0:i],axis=1))[0][0]]=50

    plt.scatter([k[0] for k in kp], [k[1] for k in kp], s , color = c)
    # plt.gcf().set_size_inches(img.shape[1], img.shape[0])

    if isinstance(save, str):
        plt.savefig(save) # , dpi = 1
        print('sift saved to' + save)

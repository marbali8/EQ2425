import os
import cv2
import matplotlib.pyplot as plt

# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
# https://docs.opencv.org/3.4.9/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html#a865f4dee68820970872c46241437f9cd

def surf(img, save = False, show = False)

    ## open image
    # img = cv2.imread(imgPath) #bgr (for cv2)
    img_rgb = img[:, :, ::-1] #rgb (for plt)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)

    ## search keypoints
    sift = cv2.SIFT_create(nfeatures = 300)
    kp, desc = sift.detectAndCompute(img, None)

    ## draw keypoints
    if save or show:

        # make sure there are no kp outside the image
        # for i, k in enumerate(kp):
        #     if k.pt[0] > img.shape[1] or k.pt[1] > img.shape[0]:
        #         print(i, k.pt)

        # aux_img = img.copy()
        # out = cv2.drawKeypoints(img, kp, aux_img)
        # cv2.imwrite('./kp.jpg', out)

        # draw them clearer
        implot = plt.imshow(img_rgb)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.scatter([k.pt[0] for k in kp], [k.pt[1] for k in kp], s = 10)
        # plt.gcf().set_size_inches(img.shape[1], img.shape[0])

        if save:
            dir = imgPath.split('.')
            path = '.'.join(dir[:-1]) + 'surf.' + dir[-1]
            plt.savefig(dir[:-1] + ) # , dpi = 1

        plt.show()

    return kp, desc

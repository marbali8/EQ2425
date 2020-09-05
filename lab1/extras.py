img1 = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_COLOR)
kp1, dp1 = siftFeatures(img1)

dp1_sorted = [x.sort() for x in dp1] # sorting the histograms

x = []
y = []
for teta in range(0, (360+15)*np.pi/180, 15*np.pi/180):

    img2 = rotateImage(img1, teta)
    kp2, dp2 = siftFeatures(img2)

    count = 0 # counting matches
    dp2_sorted = [tmp.sort() for tmp in dp2]

    for point1 in dp1:
        for point2 in dp2:
            if distance.euclidean(point1, point2) < 30:
               count += 1
               break
    y.append(count / len(kp1))
    x.append(teta)
    print(teta, count / len(kp1))

plt.plot(x,y)
plt.show()
# out = gray1
# img_1 = cv2.drawKeypoints(gray1,keypoints_1, out)

# Standard imports
from __future__ import print_function
import cv2
import numpy as np
import os

# cap = cv2.VideoCapture('/home/seth/Videos/vid3.mp4')
cap = cv2.VideoCapture('./A4.MOV')

alpha = 0.05
prev_score = 0
while cap.isOpened():
    # Read image
    ret, im = cap.read()
    #im = cv2.imread("good1.jpg", cv2.IMREAD_COLOR)
    #im = cv2.imread("good2.jpg", cv2.IMREAD_COLOR)
    # im = cv2.imread("bad0.jpg", cv2.IMREAD_COLOR)
    #im = cv2.imread("bad1.jpg", cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV);

    # define range of blue color in HSV
    lower_blue = np.array([69,60,90])
    upper_blue  = np.array([91,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((8,8),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = dilation

    ################ OpenCV's blob detection ####################
    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.01

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.8

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask=255-mask
    keypoints = detector.detect(reversemask)

    if keypoints:
        # print("found %d blobs" % len(keypoints))
        if len(keypoints) > 1:
            keypoints.sort(key=(lambda s: s.size))
            keypoints=keypoints[len(keypoints)-1:len(keypoints)]
            score = 1.
    else:
        # print("no blobs")
        score = 0

    filt_score = (1-alpha)*prev_score + alpha*score
    prev_score = filt_score
    # print("Filtered score =",filt_score)
    # Draw green circles around detected blobs
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(im,im, mask= mask)

    # cv2.imshow('masked Image',res)
    if filt_score > 0.5:
        # cv2.imshow('mask',mask)
        cv2.imshow('Frame',im_with_keypoints)
        print('Found tennis ball - prob =',filt_score)
    else:
        cv2.imshow('Frame',im)
        print('No tennis ball - prob =',filt_score)

    #cv2.imwrite('bad1_out.jpg', im_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

import os
import sys
import cv2
import glob
import sift
import imutils
import numpy as np
import scipy.spatial.distance as distance

def readImages(image_folder):
    image_names = sorted(glob.glob(os.path.join(image_folder, '*.png')))#[::-1]
    images = []

    for image_name in image_names:
        image = cv2.imread(image_name)
        images.append(image)

    return images

def matchFeatures(featuresA, featuresB):
    dist_matrix = distance.cdist(featuresA, featuresB)
    matches = []

    for i in range(len(dist_matrix)):
        trainIdx1, trainIdx2 = np.argpartition(dist_matrix[i], 2)[:2]
        distance1, distance2 = dist_matrix[i][trainIdx1], dist_matrix[i][trainIdx2]
        match1 = cv2.DMatch(i, trainIdx1, 0, distance1)
        match2 = cv2.DMatch(i, trainIdx2, 0, distance2)
        matches.append([match1, match2])

    good = []
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    return good

if __name__ == '__main__':
    image_folder = sys.argv[1]
    images = readImages(image_folder)

    while len(images)>1:
        imgR = images.pop()
        imgL = images.pop()

        grayImage = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        keypointsR, featuresR = sift.computeKeypointsAndDescriptors(grayImage)
        grayImage = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        keypointsL, featuresL = sift.computeKeypointsAndDescriptors(grayImage)
        
        allMatches = matchFeatures(featuresR, featuresL)
        
        src_pts = np.float32([ keypointsR[m.queryIdx].pt for m in allMatches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ keypointsL[m.trainIdx].pt for m in allMatches ]).reshape(-1, 1, 2)
        H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        result = cv2.warpPerspective(imgR, H, (imgR.shape[1]+imgL.shape[1], imgR.shape[0]))
        result[0:imgL.shape[0], 0:imgL.shape[1]] = imgL
        images.append(result)

    #result = imutils.resize(images[0], height=260)
    result = images[0]
    cv2.imwrite("result.jpg", result)
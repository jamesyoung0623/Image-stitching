import os
import sys
import cv2
import glob
import math
import sift
import imutils
import numpy as np
import scipy.spatial.distance as distance

def readImages(image_folder):
    #image_names = sorted(glob.glob(os.path.join(image_folder, '*.JPG')))#[::-1]
    image_names = sorted(glob.glob(os.path.join(image_folder, '*.png')))[::-1]
    #image_names = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))#[::-1]
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

def cylindricalWarpImage(img):
    f = 5500
    h, w, _ = img.shape
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

    cyl = np.zeros_like(img)
    cyl_h, cyl_w, _ = cyl.shape
    x_c = cyl_w/2.0
    y_c = cyl_h/2.0
    for x_cyl in range(cyl_w):
        for y_cyl in range(cyl_h):
            theta = (x_cyl-x_c)/f

            X = np.array([math.sin(theta), (y_cyl-y_c)/f, math.cos(theta)])
            X = np.dot(K, X)
            x_im = X[0]/X[2]
            y_im = X[1]/X[2]
            
            if x_im < 0 or x_im >= w or y_im < 0 or y_im >= h:
                continue

            cyl[y_cyl][x_cyl] = img[int(y_im), int(x_im)]

    return cyl

if __name__ == '__main__':
    image_folder = sys.argv[1]
    images = readImages(image_folder)
    images[-1] = cylindricalWarpImage(images[-1])

    while len(images)>1:
        imgL = images.pop()
        imgR = images.pop()

        imgR = cylindricalWarpImage(imgR)
        grayImage = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        keypointsR, featuresR = sift.computeKeypointsAndDescriptors(grayImage)
        
        grayImage = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        keypointsL, featuresL = sift.computeKeypointsAndDescriptors(grayImage)
        
        allMatches = matchFeatures(featuresR, featuresL)
        
        pointsR = [keypointsR[m.queryIdx].pt for m in allMatches]
        pointsL = [keypointsL[m.trainIdx].pt for m in allMatches]

        src_pts = np.float32(pointsR).reshape(-1, 1, 2)
        dst_pts = np.float32(pointsL).reshape(-1, 1, 2)
        H = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)[0]

        result = cv2.warpAffine(imgR, H, (imgR.shape[1]+imgL.shape[1], imgL.shape[0]))

        for i in range(imgL.shape[0]):
            for j in range(imgL.shape[1]):
                if not np.all(imgL[i][j] == 0):
                    result[i][j] = imgL[i][j]

        prev = -1
        for i in range(result.shape[1]):
            column = np.array([result[idx][i] for idx in range(result.shape[0])])
            if np.all(column == 0):
                if i != (prev+1):
                    result = result[:, (prev+1):i-400]
                    break
                else:
                    prev = i
        
        images.append(result)

    result = images[0][50:1400, :]
    cv2.imwrite("result.jpg", result)
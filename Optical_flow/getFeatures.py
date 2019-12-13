'''
    Identify features within the bounding box for each object using Harris corners or Shi-Tomasi features.
    Good features to track are the ones whose motion can be estimated reliably.
    use corners_shi_tomasi (in skimage)
    ----------------------
    -function: getFeatures
    -inputs: grayscale image (H*W matrix, hence one-frame),
             bounding boxes (F*4*2 matrix, F = no. of objects being tracked)
    -output: features detected (two N*F matrices, respectively for x & y coordinates, N = maximum no. of features detected in one box)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import corner_shi_tomasi

def getFeatures(img_gray, boxes):
    # print(np.shape(boxes))
    # print(boxes)
    cor1, cor2, cor3, cor4 = boxes[0] # we are testing with just the first row of boxes
    y1, x1 = cor1
    y2, x2 = cor2
    y3, x3 = cor3
    y4, x4 = cor4
    x_min, x_max = np.min([x1, x2, x3, x4]), np.max([x1, x2, x3, x4])
    y_min, y_max = np.min([y1, y2, y3, y4]), np.max([y1, y2, y3, y4])

    features = corner_shi_tomasi(img_gray, sigma=1)

    # go through anms
    max_pts = 20
    # threshold some points first
    mean = np.mean(features)
    std = np.std(features)
    features = np.where(features > mean + std, features, 0)

    # create array size of bonding box
    new_im = features[y_min:y_max, x_min:x_max]
    print(np.shape(new_im))
    cimg = new_im

    H, W = np.shape(cimg)
    filtered = np.extract(cimg != 0, cimg)
    colIndex, rowIndex = np.meshgrid(range(W), range(H))
    filtered_rowIndex = np.extract(cimg != 0, rowIndex)
    filtered_colIndex = np.extract(cimg != 0, colIndex)
    radius = np.zeros(np.shape(filtered))
    length = np.shape(filtered)[0]
    print(length)

    for curr in range(length):
        for other in range(length):
            if curr == other: continue
            if filtered[curr] / filtered[other] < 0.9: continue
            currRow = filtered_rowIndex[curr]
            currCol = filtered_colIndex[curr]
            otherRow = filtered_rowIndex[other]
            otherCol = filtered_colIndex[other]
            currDist = np.sqrt(
                (currRow - otherRow) * (currRow - otherRow) + (currCol - otherCol) * (currCol - otherCol))
            if (currDist > radius[curr]):
                radius[curr] = currDist
    combined = np.c_[radius, filtered_rowIndex, filtered_colIndex]

    combined = combined[combined[:, 0].argsort()[::-1]]

    x = combined[:, 2]
    x = x[0:max_pts]
    y = combined[:, 1]
    y = y[0:max_pts]
    # maybe
    y = y[:, np.newaxis]
    x = x[:, np.newaxis]

    x = x + x_min
    y = y + y_min
    x_return = x.astype('int32')
    y_return = y.astype('int32')


    # # showing the feature points
    # coords = np.hstack((x_return, y_return))
    # for i in range(len(coords)):
    #     x, y = coords[i][0], coords[i][1]
    #     cv2.circle(img_gray, (x, y), 6, (255, 0, 0))
    #     cv2.rectangle(img_gray, (x_min, y_min), (x_max, y_max), (255, 255, 255))

    return x_return, y_return, img_gray




def getBoundingBox(im):
    plt.imshow(im, cmap='gray')
    points = plt.ginput(4);
    plt.show()

    points = np.asarray(points)
    points = points.astype('int64')
    x, y, w, h = cv2.boundingRect(points)
    return np.array([[[y,x], [y,x+w], [y+h,x], [y+h,x+w]]])

def pointsToBox(im, pts):
    pts = pts.astype('int32')
    x, y, w, h = cv2.boundingRect(pts)
    point_a = (x, y)
    point_b = (x + w, y + h)
    color = (int(0), int(255), int(255))
    for i in range(len(pts)):
        x, y = pts[i][0], pts[i][1]
        cv2.circle(im, (x, y), 6, (0, 255, 255))

    cv2.rectangle(im, point_a, point_b, color)
    return im


def interactiveGetFeats(im):
    box = getBoundingBox(im)
    return getFeatures(im, box)


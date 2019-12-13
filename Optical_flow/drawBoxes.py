'''
    Manually draw a bounding box around an object in the first frame. I imagine how this works is that the user
    indicates at the start of the program which object(s) they want the program to track by drawing a box around them.
    -input: first frame of video. (H*W*3 matrix)
    -output: list/array of bounding boxes containing coordinates of their four corners?
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random as rng

rng.seed(1245)

e_t1 = cv2.imread('ellipse_t1.jpg')
# e_t1 = cv2.cvtColor(e_t1, cv2.COLOR_BGR2GRAY)
frame0 = cv2.imread('./data/frame0.jpg')

plt.imshow(frame0, cmap = 'gray')
points = plt.ginput(4);
plt.show()

points = np.asarray(points)
points = points.astype('int64')

print(points)

x, y, w, h = cv2.boundingRect(points)


point_a = (x,y)
point_b = (x+w, y+h)
# color = (int(0), rng.randint(0,256), rng.randint(0,256))
color = (int(255), int(255), int(255))
cv2.rectangle(frame0, point_a, point_b, color)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', frame0)


cv2.waitKey()
cv2.destroyAllWindows()
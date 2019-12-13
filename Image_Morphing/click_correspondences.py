'''
  File name: click_correspondences.py
  Author: haorongy
  Date created: 
'''
from helpers import cpselect
from PIL import Image
import numpy as np

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''


def click_correspondences(im1, im2):
  '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''

  img1 = np.array(im1, dtype='uint8')
  img2 = np.array(im2, dtype='uint8')

  resize_img1 = np.array(Image.fromarray(img1).resize((300, 300)))
  resize_img2 = np.array(Image.fromarray(img2).resize((300, 300)))

  im1_pts, im2_pts = cpselect(resize_img1, resize_img2)

  arr_borders = [[0, 0],
                  [150, 0],
                  [299, 0],
                  [0, 150],
                  [0, 299],
                  [150, 299],
                  [299, 299],
                  [299, 150]]

  im1_pts = np.concatenate((im1_pts, arr_borders), axis=0)
  im2_pts = np.concatenate((im2_pts, arr_borders), axis=0)
  
  return im1_pts, im2_pts

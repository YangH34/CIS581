'''
  File name: carv.py
  Author:
  Date created:
'''

'''
  File clarification:
    Aimed to handle finding seams of minimum energy, and seam removal, the algorithm
    shall tackle resizing images when it may be required to remove more than one seam, 
    sequentially and potentially along different directions.
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT nr: the numbers of rows to be removed from the image.
    - INPUT nc: the numbers of columns to be removed from the image.
    - OUTPUT Ic: (n − nr) × (m − nc) × 3 matrix representing the carved image.
    - OUTPUT T: (nr + 1) × (nc + 1) matrix representing the transport map.
'''

import numpy as np
from rmVerSeam import rmVerSeam
from rmHorSeam import rmHorSeam
from genEngMap import genEngMap
from cumMinEngHor import cumMinEngHor
from cumMinEngVer import cumMinEngVer
import os
from PIL import Image

def carv(I, nr, nc):
  #for saving images
  directory = 'Images'
  parent_dir = os.getcwd()
  path = os.path.join(parent_dir, directory)

  T = np.empty((nr, nc), dtype='object')
  clone = I.copy()

  #let 0 denote from left, 1 denote from up
  #initialize the first row and col
  tempClone = I.copy()
  for j in range(1, nc):
    tempClone, e = getTupleLeft(tempClone)
    T[0][j] = (tempClone, e, 0)

  tempClone = I.copy()
  for i in range(1, nr):
    tempClone, e = getTupleUp(tempClone)
    T[i][0] = (tempClone, e, 1)

  # fill in the rest of row and cols
  for i in range(1, nr):
    for j in range(1, nc):
      e_up = T[i - 1][j][1]
      e_left = T[i][j - 1][1]

      if e_up <= e_left:
        temp, e_temp = getTupleUp(T[i - 1][j][0])
        T[i][j] = (temp, e_temp, 1)
      else:
        temp, e_temp = getTupleLeft(T[i][j - 1][0])
        T[i][j] = (temp, e_temp, 0)


  trace_list = []
  i = nr - 1
  j = nc - 1
  while not(i == 0 and j == 0):
    dir = T[i][j][2]
    trace_list.append(dir)
    if dir == 0:
      j = j - 1
    elif dir == 1:
      i = i - 1
  trace_list.reverse()

  count = 100
  for i in trace_list:
    if i == 0:
      clone, dummy = getTupleLeft(clone)

      im = pad(clone, I)
      im.save(path + '/im_seam' + str(count) + '.png')
      count = count + 1
    else:
      clone, dummy = getTupleUp(clone)
      im = pad(clone, I)
      im.save(path + '/im_seam' + str(count) + '.png')
      count = count + 1

  Ic = clone
  return Ic, T


def getTupleUp(clone):
  dataEn = genEngMap(clone)
  My, Tby = cumMinEngHor(dataEn)
  clone2, e = rmHorSeam(clone, My, Tby)
  return clone2, e

def getTupleLeft(clone):
  dataEn = genEngMap(clone)
  Mx, Tbx = cumMinEngVer(dataEn)
  clone2, e = rmVerSeam(clone, Mx, Tbx)
  return clone2, e

def pad(clone, I):
  n, m, z = np.shape(I)
  cn, cm, z = np.shape(clone)
  blank_guy = np.zeros((n, m, z), dtype='uint8')
  blank_guy.fill(255)
  for x in range(z):
    for i in range(cn):
      for j in range(cm):
        blank_guy[i][j][x] = clone[i][j][x]
  im = Image.fromarray(blank_guy)
  return im

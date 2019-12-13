'''
  File name: rmVerSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes vertical seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - INPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
    - OUTPUT Ix: n × (m - 1) × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
import numpy as np

def rmVerSeam(I, Mx, Tbx):
  n, m, z = np.shape(I)
  clone = I.copy()
  outImg = np.zeros((n, m - 1, z), dtype="float64")
  minVal = Mx[n - 1][0]
  index = 0

  for i in range(0, m):
    if Mx[n - 1][i] < minVal:
      minVal = Mx[n - 1][i]
      index = i

  currY = n - 1
  currX = index
  for i in range(0, n):
    flag = 0
    for j in range(0, m):
      if j != currX:
        for x in range(3):
          outImg[currY][j + flag][x] = clone[currY][j][x]
      else:
        flag = -1
    currX = currX + int(Tbx[currY][currX])
    currY = currY - 1

  outImg = outImg.astype(np.uint8)

  Ix = outImg
  E = minVal
  return Ix, E

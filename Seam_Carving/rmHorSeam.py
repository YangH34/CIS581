'''
  File name: rmHorSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes horizontal seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - INPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
    - OUTPUT Iy: (n − 1) × m × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''

import numpy as np

def rmHorSeam(I, My, Tby):
  n, m, z = np.shape(I)
  clone = I.copy()
  outImg = np.zeros((n - 1, m, z), dtype="float64")
  minVal = My[0][m - 1]
  index = 0

  for i in range(0, n):
    if My[i][m - 1] < minVal:
      minVal = My[i][m - 1]
      index = i

  currX = m - 1
  currY = index
  for j in range(0, m):
    flag = 0
    for i in range(0, n):
      if i != currY:
        for x in range(3):
          outImg[i + flag][currX][x] = clone[i][currX][x]
      else:
        flag = -1
    currY = currY + int(Tby[currY][currX])
    currX = currX - 1

  outImg = outImg.astype(np.uint8)

  Iy = outImg
  E = minVal
  return Iy, E

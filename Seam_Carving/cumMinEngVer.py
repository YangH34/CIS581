'''
  File name: cumMinEngVer.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the vertical seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - OUTPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
'''
import numpy as np

def cumMinEngVer(e):
  n, m = np.shape(e)
  # print(n, m)
  clone = e.copy()
  trace = np.zeros(np.shape(e),dtype="int32")
  # print(clone)

  for i in range(1, n):
    for j in range(0, m):

      if j == 0:
        #update cumulative energy
        tmid = clone[i - 1][j]
        tright = clone[i - 1][j + 1]
        acc = min(tmid, tright)
        clone[i][j] = clone[i][j] + acc
        #update traceback
        if acc == tmid:
          trace[i][j] = 0
        elif acc == tright:
          trace[i][j] = 1

      elif j == m - 1:
        # update cumulative energy
        tmid = clone[i - 1][j]
        tleft = clone[i - 1][j - 1]
        acc = min(tmid, tleft)
        clone[i][j] = clone[i][j] + acc
        # update traceback
        if acc == tmid:
          trace[i][j] = 0
        elif acc == tleft:
          trace[i][j] = -1
      else:
        tleft = clone[i - 1][j - 1]
        tmid = clone[i - 1][j]
        tright = clone[i - 1][j + 1]
        acc = min(tleft, tmid, tright)
        clone[i][j] = clone[i][j] + acc
        # update traceback
        if acc == tleft:
          trace[i][j] = -1
        elif acc == tmid:
          trace[i][j] = 0
        elif acc == tright:
          trace[i][j] = 1

  Mx = clone
  Tbx = trace
  return Mx, Tbx
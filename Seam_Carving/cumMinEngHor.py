'''
  File name: cumMinEngHor.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the horizontal seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - OUTPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
'''
import numpy as np

def cumMinEngHor(e):
  n, m = np.shape(e)
  clone = e.copy()
  trace = np.zeros(np.shape(e))

  for j in range(1, m):
    for i in range(0, n):

      if i == 0:
        # update cumulative energy
        lmid = clone[i][j - 1]
        ldown = clone[i + 1][j - 1]
        acc = min(lmid, ldown)
        clone[i][j] = clone[i][j] + acc
        # update traceback
        if acc == lmid:
          trace[i][j] = 0
        elif acc == ldown:
          trace[i][j] = 1

      elif i == n - 1:
        # update cumulative energy
        lmid = clone[i][j - 1]
        lup = clone[i - 1][j - 1]
        acc = min(lmid, lup)
        clone[i][j] = clone[i][j] + acc
        # update traceback
        if acc == lmid:
          trace[i][j] = 0
        elif acc == lup:
          trace[i][j] = -1
      else:
        lup = clone[i - 1][j - 1]
        lmid = clone[i][j - 1]
        ldown = clone[i + 1][j - 1]
        acc = min(lup, lmid, ldown)
        clone[i][j] = clone[i][j] + acc
        # update traceback
        if acc == lup:
          trace[i][j] = -1
        elif acc == lmid:
          trace[i][j] = 0
        elif acc == ldown:
          trace[i][j] = 1

  My = clone
  Tby = trace
  return My, Tby
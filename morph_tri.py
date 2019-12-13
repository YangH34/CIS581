'''
  File name: morph_tri.py
  Author: haorongy
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import numpy as np
import os
from helpers import interp2
from PIL import Image

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  im1 = np.array(Image.fromarray(im1.astype(np.uint8)).resize((300, 300)))
  im2 = np.array(Image.fromarray(im2.astype(np.uint8)).resize((300, 300)))

  # use meshgrid to obtain arr (shape (90000, )) of corresponding trig for all points in image
  xx, yy = np.meshgrid(range(300), range(300))
  yy_horiz = yy.reshape(1 , 300 * 300)
  xx_horiz = xx.reshape(1 , 300 * 300)
  xy_combine = np.concatenate((xx_horiz.T, yy_horiz.T), axis=1)
  cloneImg = im1.copy()
  resultImg = np.zeros((warp_frac.size, 300, 300, 3))

  #getimage
  for i in range(warp_frac.size):
    # compute average triangulation with warp_frac and optain its delaunay
    pts_avg = im2_pts * warp_frac[i] + im1_pts * (1 - warp_frac[i])
    tri = Delaunay(pts_avg)
    tri_arr = tri.find_simplex(xy_combine)

    im1_morph_r, im1_morph_g, im1_morph_b = findStuff(xx, yy, im1, pts_avg, im1_pts, tri, tri_arr)
    im2_morph_r, im2_morph_g, im2_morph_b = findStuff(xx, yy, im2, pts_avg, im2_pts, tri, tri_arr)

    cloneImg[:, :, 0] = im2_morph_r * dissolve_frac[i] + im1_morph_r * (1 - dissolve_frac[i])
    cloneImg[:, :, 1] = im2_morph_g * dissolve_frac[i] + im1_morph_g * (1 - dissolve_frac[i])
    cloneImg[:, :, 2] = im2_morph_b * dissolve_frac[i] + im1_morph_b * (1 - dissolve_frac[i])

    resultImg[i, :, :, :] = cloneImg

  return resultImg

def findStuff(xx, yy, img, pts_avg, pts_orig, tri, tri_arr):
  corr_tri = tri_arr[yy * 300 + xx]
  trig_pts_avg = pts_avg[tri.simplices]
  trig_pts_orig = pts_orig[tri.simplices]

  Ax_avg = trig_pts_avg[corr_tri, 0, 0]
  Bx_avg = trig_pts_avg[corr_tri, 1, 0]
  Cx_avg = trig_pts_avg[corr_tri, 2, 0]
  Ay_avg = trig_pts_avg[corr_tri, 0, 1]
  By_avg = trig_pts_avg[corr_tri, 1, 1]
  Cy_avg = trig_pts_avg[corr_tri, 2, 1]

  Ax_orig = trig_pts_orig[corr_tri, 0, 0]
  Bx_orig = trig_pts_orig[corr_tri, 1, 0]
  Cx_orig = trig_pts_orig[corr_tri, 2, 0]
  Ay_orig = trig_pts_orig[corr_tri, 0, 1]
  By_orig = trig_pts_orig[corr_tri, 1, 1]
  Cy_orig = trig_pts_orig[corr_tri, 2, 1]

  ones = np.zeros((300, 300))
  ones.fill(1)
  ones_horiz = np.zeros((1, 90000))
  ones_horiz.fill(1)

  Ax_avg_flat = Ax_avg.flatten()
  Bx_avg_flat = Bx_avg.flatten()
  Cx_avg_flat = Cx_avg.flatten()
  Ay_avg_flat = Ay_avg.flatten()
  By_avg_flat = By_avg.flatten()
  Cy_avg_flat = Cy_avg.flatten()

  Ax_orig_flat = Ax_orig.flatten()
  Bx_orig_flat = Bx_orig.flatten()
  Cx_orig_flat = Cx_orig.flatten()
  Ay_orig_flat = Ay_orig.flatten()
  By_orig_flat = By_orig.flatten()
  Cy_orig_flat = Cy_orig.flatten()

  # big fat chunk to get Avg into 90000 * 3 * 3 matrix
  tempABCx = np.vstack((Ax_avg_flat, Bx_avg_flat, Cx_avg_flat))
  tempx = tempABCx.T
  tempx = np.vsplit(tempx, 90000)
  tempx = np.hstack(tempx)
  tempx = np.squeeze(tempx, axis=0)
  tempABCy = np.vstack((Ay_avg_flat, By_avg_flat, Cy_avg_flat))
  tempy = tempABCy.T
  tempy = np.vsplit(tempy, 90000)
  tempy = np.hstack(tempy)
  tempy = np.squeeze(tempy, axis=0)
  ones_flat = np.zeros(270000, dtype='float64')
  ones_flat.fill(1)
  temp_avg = np.vstack((tempx, tempy, ones_flat))
  temp_avg = np.hsplit(temp_avg, 90000)
  temp_avg = np.vstack(temp_avg)
  temp_avg = np.vsplit(temp_avg, 90000)

  # big fat chunk to get orig into 90000 * 3 * 3 matrix
  tempABCxo = np.vstack((Ax_orig_flat, Bx_orig_flat, Cx_orig_flat))
  tempxo = tempABCxo.T
  tempxo = np.vsplit(tempxo, 90000)
  tempxo = np.hstack(tempxo)
  tempxo = np.squeeze(tempxo, axis=0)
  tempABCyo = np.vstack((Ay_orig_flat, By_orig_flat, Cy_orig_flat))
  tempyo = tempABCyo.T
  tempyo = np.vsplit(tempyo, 90000)
  tempyo = np.hstack(tempyo)
  tempyo = np.squeeze(tempyo, axis=0)
  temp_o = np.vstack((tempxo, tempyo, ones_flat))
  temp_o = np.hsplit(temp_o, 90000)
  temp_o = np.vstack(temp_o)
  temp_o = np.vsplit(temp_o, 90000)

  xx_flat = xx.flatten()
  yy_flat = yy.flatten()
  ones_flat_2 = np.zeros(90000, dtype='float64')
  ones_flat_2.fill(1)
  xy1_stack = np.vstack((xx_flat, yy_flat, ones_flat_2))
  xy1 = np.hsplit(xy1_stack, 90000)
  xy1 = np.vstack(xy1)
  xy1 = np.vsplit(xy1, 90000)

  solutiontMtx = np.linalg.solve(temp_avg, xy1)

  solution_pts = np.matmul(temp_o, solutiontMtx)
  solution_pts = np.squeeze(solution_pts, axis=2)
  solution_pts = np.hsplit(solution_pts, 3)
  sol_x = solution_pts[0]
  sol_y = solution_pts[1]
  sol_x = sol_x.flatten()
  sol_y = sol_y.flatten()
  sol_x_r = sol_x.reshape((300, 300))
  sol_y_r = sol_y.reshape((300, 300))

  img_r = img[:, :, 0]
  img_g = img[:, :, 1]
  img_b = img[:, :, 2]
  im1_morph_r = interp2(img_r, sol_x_r, sol_y_r)
  im1_morph_g = interp2(img_g, sol_x_r, sol_y_r)
  im1_morph_b = interp2(img_b, sol_x_r, sol_y_r)

  return im1_morph_r, im1_morph_g, im1_morph_b

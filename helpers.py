'''
  File name: helpers.py
  Author:
  Date created:
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os



'''
  File clarification:
    Helpers file that contributes the project
    You can design any helper function in this file to improve algorithm
'''

class cpselect_recorder:
	def __init__(self, img1,img2):

		fig, (self.Ax0, self.Ax1) = plt.subplots(1, 2, figsize = (20, 20))

		self.Ax0.imshow(img1)
		self.Ax0.axis('off')

		self.Ax1.imshow(img2)
		self.Ax1.axis('off')

		fig.canvas.mpl_connect('button_press_event', self)
		self.left_x = []
		self.left_y = []
		self.right_x = []
		self.right_y = []

	def __call__(self, event):
		circle = plt.Circle((event.xdata, event.ydata),color='r')
		if event.inaxes == self.Ax0:
			self.left_x.append(event.xdata)
			self.left_y.append(event.ydata)
			self.Ax0.add_artist(circle)
			plt.show()
		elif event.inaxes == self.Ax1:
			self.right_x.append(event.xdata)
			self.right_y.append(event.ydata)
			self.Ax1.add_artist(circle)
			plt.show()

def cpselect(img1,img2):
	resize_img1 = np.array(Image.fromarray(img1).resize((300, 300)))
	resize_img2 = np.array(Image.fromarray(img2).resize((300, 300)))
	point = cpselect_recorder(resize_img1,resize_img2)
	plt.show()
	point_left = np.concatenate([(np.array(point.left_x)*img1.shape[1]*1.0/300)[...,np.newaxis],\
								(np.array(point.left_y)*img1.shape[0]*1.0/300)[...,np.newaxis]],axis = 1)
	point_right = np.concatenate([(np.array(point.right_x)*img2.shape[1]*1.0/300)[...,np.newaxis],\
								(np.array(point.right_y)*img2.shape[0]*1.0/300)[...,np.newaxis]],axis = 1)
	plt.scatter(point_left[:,0], point_left[:,1])
	plt.imshow(img1)
	#plt.show()
	plt.scatter(point_right[:,0], point_right[:,1])
	plt.imshow(img2)
	#plt.show()
	return point_left, point_right

#   Function Input
#   v     M*N            the value lies on grid point which is corresponding to the meshgrid coordinates
#   xq    M1*N1 or M2    the query points x coordinates
#   yq    M1*N1 or M2    the query points y coordinates
#
##########
#   Function Output
#   interpv , the interpolated value at querying coordinates xq, yq, it has the same size as xq and yq.
##########
#   For project 1, v = Mag
#   xq and yq are the coordinates of the interpolated location,
#   i.e the coordinates computed based on the gradient orientation.

def interp2(v, xq, yq):
	dim_input = 1
	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise Exception('query coordinates Xq Yq should have same shape')

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val

def video_create(path, name, parent_dir, fps):
	image_folder = path
	video_name = parent_dir + '/videos/' + name + '_video.avi'

	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

	for image in images:
		video.write(cv2.imread(os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()


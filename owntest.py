from click_correspondences import click_correspondences
from PIL import Image
import numpy as np
from morph_tri import morph_tri
import os
from helpers import video_create

#please enter names of your images below, for example my images are 'cat2.jpg' and 'dog.jpg'
startImage = 'cat_crop.jpg'
endImage = 'dog_crop.jpg'

#please enter the number of frames you want, and the fps. default is 36 frames with fps of 18
frames = 10
fps = 10

img1 = Image.open(startImage)
img2 = Image.open(endImage)
directory = startImage + '_to_' + endImage
parent_dir = os.getcwd()
path = os.path.join(parent_dir, directory)
os.mkdir(path)


point_img1, point_img2 = click_correspondences(img1, img2)

img1 = np.array(img1, dtype='uint8')
img2 = np.array(img2, dtype='uint8')


warp_frac = np.linspace(0, 1.0, frames, endpoint=True)
dissolve_frac = warp_frac
resultImageArr = morph_tri(img1, img2, point_img1, point_img2, warp_frac, dissolve_frac)

a = np.shape(resultImageArr)[0]
for i in range(a):
    cloneImg = np.zeros((300, 300, 3), dtype='uint8')
    currImage = resultImageArr[i]

    cloneImg[:, :, :] = currImage[:, :, :]
    im = Image.fromarray(cloneImg)
    if i < 10:
        im.save(path + '/im_morph0' + str(i) + '.png')
    else:
        im.save(path + '/im_morph' + str(i) + '.png')

video_create(path, directory, parent_dir, fps)

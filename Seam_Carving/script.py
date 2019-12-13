import matplotlib.pyplot as plt
import numpy as np
from carv import carv
from helpers import video_create
import os

#please insert image name and desired video fps, nr and nc below
Image = 'mountains.jpg'
fps = 8
nr = 3
nc = 3

directory = 'Images'
parent_dir = os.getcwd()
path = os.path.join(parent_dir, directory)

img1 = plt.imread(Image)
img1 = np.asarray(img1, dtype="uint8")

img1 = carv(img1, nr, nc)

video_create(path, Image, parent_dir, fps)
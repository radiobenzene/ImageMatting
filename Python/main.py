from cv2 import *
from functions import *
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

# Calling image object
img_obj = initializeVariables()

name = "GT06"

os.path.join("Images","input_training_lowres","{}.png".format(name))

image = np.array(Image.open(os.path.join("Images","input_training_lowres","{}.png".format(name))))
image_trimap = np.array(ImageOps.grayscale(Image.open(os.path.join("Images","trimap_training_lowres", "Trimap2", "{}.png".format(name)))))

#alpha,pixel_count = getBayesianMatte(image,image_trimap, img_obj.N, img_obj.sigma, img_obj.min_N) 
alpha,pixel_count = getBayesianMatte(image, image_trimap,  name, img_obj.N, img_obj.sigma, img_obj.min_N) 
alpha_int8 = np.array(alpha,dtype = int)

displayImage('Title', alpha)

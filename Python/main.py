from cv2 import *
from functions import *

# Reading image
img = cv2.imread("Images/input_training_lowres/GT05.png")

# Reading trimap
trimap = cv2.imread("Images/trimap_training_lowres/Trimap1/GT05.png")

# Reading Ground Truth image
GT_img = cv2.imread("Images/gt_training_lowres/GT05.png")

# Converting images to double
img = img.astype(float) / 255.0
trimap = trimap.astype(float) / 255.0

# Calling image object
img_obj = initializeVariables()

[height, width, c] = img.shape
A = getMSE(1, 2)
print(A)

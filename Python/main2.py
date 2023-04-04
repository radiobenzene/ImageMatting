from cv2 import *
from functions2 import *
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# n = len(sys.argv)
# image_name = sys.argv[1]


# main(img_name = "GT01", user_selector = False, save = False, show_FG = False, compositing = False)
main(user_selector = False, save = True, show_FG = True, compositing= True)
# main(user_selector = True, save = True, show_FG = True, compositing= True)

# execution_longSeq()
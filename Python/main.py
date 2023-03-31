from cv2 import *
from functions import *
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import sys

n = len(sys.argv)
image_name = sys.argv[1]

main(image_name)

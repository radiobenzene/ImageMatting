# File to store functions
import cv2
import math
import numpy as np
from scipy.signal import gaussian
from skimage.morphology import square
from skimage.measure import compare_mse

"""
    Function to get Bayesian Matte
    Params:
        img - image
        trimap - trimap
        img_obj - image object containing parameters
"""
def getBayesianMatte(img, trimap, img_obj):
    
    # Setting masks for foreground, background and unknown
    background_mask = (trimap == 0) # Background where trimap values = 0
    foreground_mask = (trimap == 1) # Foreground where trimap values = 1
    unknown_area_mask = ~(background_mask | foreground_mask) # If neither, then unknown
    
    # Initializing Foreground
    F = img.copy()
    F[~foreground_mask.repeat(3).reshape(foreground_mask.shape)] = 0

    # Initializing Background
    B = img.copy()
    B[~background_mask.repeat(3).reshape(background_mask.shape)] = 0
    
    # Initializing Alpha channel
    alpha_channel = np.zeros(trimap.shape)
    alpha_channel[foreground_mask] = 1
    alpha_channel[unknown_area_mask] = np.nan
    
    # Initializing unknown points
    unknown_points = np.sum(unknown_area_mask)
    
    # Gaussian Weighting parameter
    gaussian_weighting = gaussian(img_obj.N, img_obj.sigma).reshape(-1, 1) * gaussian(img_obj.N, img_obj.sigma).reshape(1, -1)
    
    # Normalizing gaussian weighting
    gaussian_weighting = gaussian_weighting / np.max(gaussian_weighting)
    
    # square structuring element for eroding the unknown region(s)
    se = square(3)
    
    n = 1
    unknown_region=unknown_area_mask
    iter = 1


"""
Function to run a window
Params: 
    img_area - image
    x, y - pixel values
    N - window size
"""
def runWindow(img_area, x, y, N):
    # Get dimensions of the image
    [height, width, c] = img_area.shape
    
    # Initializing window size
    half_win_size = math.floor(N / 2)
    N_1 = half_win_size
    N_2 = N - half_win_size - 1
    
    # Initializing window value
    window_val = np.empty((N, N, c))
    window_val[:] = np.nan
    
    # Setting min and max values for x-axis
    x_min = max(1, x - N_1)
    x_max = min(width, x + N_2)
    
    # Setting min and max values for y-axis
    y_min = max(1, y - N_1)
    y_max = min(height, y + N_2)
    
    # Getting pixel values for x coordinates (min and max)
    pixel_x_min = half_win_size - (x - x_min) + 1 
    pixel_x_max = half_win_size + (x_max - x) + 1
    
    # Getting pixel values for y coordinates (min and max)
    pixel_y_min = half_win_size - (y - y_min) + 1 
    pixel_y_max = half_win_size + (y_max - y) + 1
    
    # Setting window values
    window_val[pixel_y_min:pixel_y_max, pixel_x_min:pixel_x_max, :] = img_area[y_min:y_max, x_min:x_max, :]
    
# Defining class to initialize variables
class initializeVariables:
    # Initialize Window Size
    N = 120
    
    # Initialize Variance for Gaussian weighting
    sigma = 50
    
    # Initialize camera variance
    cam_sigma = 0.05
    
    # Initialize Minimum window size
    min_N = 10
    
    # Initialize Clustering variance
    clustering_variance = 0.90
    
    """ 
        Function to calculate MSE
        @FerniA
    """
def getMSE(alpha_val, gt_img):
    gt_img = np.double(gt_img)
    gt_img = gt_img[:, :, 0]
    mse_val = compare_mse(alpha_val, gt_img)
    return mse_val
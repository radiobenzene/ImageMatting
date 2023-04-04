# File to store functions
import cv2
import math
import numpy as np
from scipy.signal import gaussian
from skimage.morphology import square
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import os
from tqdm import tqdm
import time
from numba import jit 
from tkinter import filedialog
"""
Function to display Image
Params:
    title: Image Title
    img: Display Image
Returns:
    Image box with a title 
"""

def displayImage(title, img):
    #plt.show(img)
    #plt.title(title)
   # plt.show()
    
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
    Function to create a MATLAB style fspecial('gaussian')
    Params: 
        shape: shape of the window
        variance: Variance of the Gaussian distribution
    Returns:
        Gaussian distribution with sigma 
"""

def fspecial(shape, variance):
    
    # Setting N_1 and N_2
    N_1, N_2 = [(index - 1.)/2. for index in shape]
    
    y, x = np.ogrid[-N_1:N_1+1, -N_2:N_2+1]
    
    gaussian_window = np.exp(-(x*x + y*y)/(2.* variance*variance))
    gaussian_window[gaussian_window < np.finfo(gaussian_window.dtype).eps*gaussian_window.max()] = 0
    
    sumh = gaussian_window.sum()
    if sumh != 0:
        gaussian_window /= sumh
        
    return gaussian_window

'''
    Class Node for Orchard Boumann clustering
'''
class Node(object):

    def __init__(self, matrix, w):
        W = np.sum(w)
        self.w = w
        self.X = matrix
        self.left = None
        self.right = None
        self.mu = np.einsum('ij,i->j', self.X, w)/W
        diff = self.X - np.tile(self.mu, [(self.X.shape)[0], 1])
        t = np.einsum('ij,i->ij', diff, np.sqrt(w))
        self.cov = (t.T @ t)/W + 1e-5*np.eye(3)
        self.N = self.X.shape[0]
        V, D = np.linalg.eig(self.cov)
        self.lmbda = np.max(np.abs(V))
        self.e = D[np.argmax(np.abs(V))]


'''
    Function to perform Orchard Boumann Clustering
    Params:
        S: Pixel values
        w: Weight values
        min_Var: Minimum Variance for splitting
    Returns:
        mu: Mean
        sigma: Variance
'''

def clusterElements(S, w, min_var=0.05):
    mu, sigma = [], []
    nodes = []
    nodes.append(Node(S, w))

    while max(nodes, key=lambda x: x.lmbda).lmbda > min_var:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        mu.append(node.mu)
        sigma.append(node.cov)

    return np.array(mu), np.array(sigma)

'''
    Function to split a node
    
'''

def split(nodes):
    idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
    C_i = nodes[idx_max]
    idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)
    C_a = Node(C_i.X[idx], C_i.w[idx])
    C_b = Node(C_i.X[np.logical_not(idx)], C_i.w[np.logical_not(idx)])
    nodes.pop(idx_max)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes


'''
Function for converting image to float and normalization
Params:
    img: Input image
Returns:
    normalized image
'''

def convertImage(img):
    img = np.array(img,dtype = 'float')
    img /= 255
    return img


'''
Function to get Bayesian matte
    Params:
        img: Input image
        trimap: Trimap image
        name: Image name
        N: default window size of 25
        variance: Gaussian sigma
        minN: minimum Neighbours
    Returns:
        the alpha matte
    
'''
def getBayesianMatte(img, trimap, name, N = 25,variance = 8,min_N = 10):
    image_trimap = np.array(ImageOps.grayscale(Image.open(os.path.join("Python","Images","trimap_training_lowres","Trimap2", "{}".format(name)))))

    
    # Converting image to float type
    img = convertImage(img)
    trimap = convertImage(trimap)
    
    # Getting dimensions of the image
    h,w,c = img.shape
    
    # Initializing Gaussian weighting
    gaussian_weights = fspecial((N,N),variance)
    gaussian_weights /= np.max(gaussian_weights)

    # Initializing Foreground objects
    foreground_map = trimap == 1
    foreground_img = np.zeros((h,w,c))
    foreground_img = img * np.reshape(foreground_map,(h,w,1))

    # Initializig Background objects
    background_map = trimap == 0
    background_img = np.zeros((h,w,c))
    background_img = img * np.reshape(background_map,(h,w,1))
    
    # Initializing Alpha channel
    unknown_map = np.logical_or(foreground_map,background_map) == False
    a_channel = np.zeros(unknown_map.shape)
    a_channel[foreground_map] = 1
    a_channel[unknown_map] = np.nan

    # Calculating total number of unknown points
    unknown_points = np.sum(unknown_map)

    # Build list for unknown points
    A,B = np.where(unknown_map == True)
    list_not_visited_points = np.vstack((A,B,np.zeros(A.shape))).T

    print("Processing Image")
    #for i in tqdm(range(100), desc="Generating Matte", ascii=False, ncols=75):

    # Solving for unknown points
    while(sum(list_not_visited_points[:,2]) != unknown_points):
        
        last_n = sum(list_not_visited_points[:,2])

        for i in range(unknown_points): 
            
            # Checking if points are in array
            if list_not_visited_points[i,2] == 1:
                continue
            
            else:
                
                # Get location of unknown points
                y,x = map(int,list_not_visited_points[i,:2])
                
                # Running window through alpha channel
                alpha_window = runWindow(a_channel[:, :, np.newaxis], x, y, N)[:,:,0]
                
                # Creating a window and weights of solved foreground window
                foreground_window = runWindow(foreground_img,x,y,N)
                foreground_weights = np.reshape(alpha_window**2 * gaussian_weights,-1)
                
                values_to_keep = np.nan_to_num(foreground_weights) > 0
                
                # Restructuring Foreground pixels
                foreground_pixels = np.reshape(foreground_window,(-1,3))[values_to_keep,:]
                foreground_weights = foreground_weights[values_to_keep]
        
                # Creating a window and weights of solved background window
                background_window = runWindow(background_img,x,y,N)
                background_weights = np.reshape((1-alpha_window)**2 * gaussian_weights,-1)
                
                values_to_keep = np.nan_to_num(background_weights) > 0
                
                # Restructuring Background pixels
                background_pixels = np.reshape(background_window,(-1,3))[values_to_keep,:]
                background_weights = background_weights[values_to_keep]
                
                # We come back to this pixel later if it doesnt has enough solved pixels around it.
                if len(background_weights) < min_N or len(foreground_weights) < min_N:
                    continue
                
                # If enough pixels, we cluster these pixels to get clustered colour centers and their covariance    matrices
                foreground_mean, foreground_covariance = clusterElements(foreground_pixels,foreground_weights)
                background_mean, background_covariance = clusterElements(background_pixels,background_weights)
                alpha_init = np.nanmean(alpha_window.ravel())
                
                # Solving for Foreground, Background and Alpha
                calculated_foreground,calculated_background,calculated_alpha = solve(foreground_mean, 
                                                                                     foreground_covariance,
                                                                                     background_mean,
                                                                                     background_covariance,
                                                                                     img[y,x],
                                                                                     0.7,
                                                                                     alpha_init)

                # Updating Foreground
                foreground_img[y, x] = calculated_foreground.ravel()
                
                # Updating Background
                background_img[y, x] = calculated_background.ravel()
                
                # Updating Alpha
                a_channel[y, x] = calculated_alpha
                
                list_not_visited_points[i,2] = 1
                if(np.sum(list_not_visited_points[:,2]) % 2000 == 0):
                    print("Processing image to generate matte ...")
                    pass

        if sum(list_not_visited_points[:,2]) == last_n:
            # Increasing window size
            N += 2
            
            # Increasing variance
            variance += 1 
            
            # Applying Gaussian weighting 
            gaussian_weights = fspecial((N,N),variance)
            gaussian_weights /= np.max(gaussian_weights)
    return a_channel

"""
Function to run a window
Params: 
    img_area - image
    x, y - pixel values
    N - window size
"""
def runWindow(img_area, x, y, N):
    
    # Get image dimensions 
    height, width, channels = img_area.shape
    
    # Get central element
    centre = N // 2 
    
    # Initialize window
    window = np.zeros((N,N,channels))      

    # Setting x_min and x_max
    x_min = max(0,x - centre)
    x_max = min(width, x + centre + 1)
    
    # Setting y_min and y_max
    y_min = max(0, y - centre)
    y_max = min(height, y + centre + 1)

    window[centre - (y - y_min):centre +(y_max - y),centre - (x - x_min):centre +(x_max-x)] = img_area[y_min:y_max,x_min:x_max]

    return window

# Defining class to initialize variables
class initializeVariables:
    # Initialize Window Size
    N = 121 #25 #120
    
    # Initialize Variance for Gaussian weighting
    sigma = 8; ###0.5
    
    # Initialize camera variance
    cam_sigma = 0.05
    
    # Initialize Minimum window size
    min_N = 20 #10
    
    # Initialize Clustering variance
    clustering_variance = 0.05

'''
    Function to solve the Maximum Likelihood problem
    Params:
        mu_F: Mean of Foreground pixel
        sigma_F: Covariance of Foreground pixel
        mu_B: Mean of Background pixel
        sigma_B: Covariance of Background pixel
        C: Current pixel
        sigma_C: pixel variance
        alpha_init: Initial value of alpha
        maxIter: Iterations to solve the value of the pixel, default value = 50
        minLike: Minimum Likeihood to reach to stop the maxIterations, default value = 1e-6 
'''

def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_init, maxIter = 50, minLike = 1e-6):

    # Initializing Matrices
    I = np.eye(3)
    
    # Initializing estimates for Foreground, Background and alpha
    fg_best = np.zeros(3)
    bg_best = np.zeros(3)
    a_best = np.zeros(1)
    
    # Initializing maximum likelihood
    maxlike = -np.inf
    
    # Initializing imverse variance
    invsgma2 = 1 / Sigma_C**2
    
    for i in range(mu_F.shape[0]):
        
        # Iterating through all values of mean and covariance
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])

        for j in range(mu_B.shape[0]):
            
            # Iterating through all values of mean and covariance
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            # Setting alpha values
            alpha = alpha_init
            
            # Initializing iterator
            myiter = 1
            
            # Initializing last likelihood
            lastLike = -1.7977e+308

            while True:
                
                # Solving the equation: Ax = b, where x has 3 values - RGB
                A = np.zeros((6,6))
                A[:3,:3] = invSigma_Fi + I*alpha**2 * invsgma2
                A[:3,3:] = A[3:,:3] = I*alpha*(1-alpha) * invsgma2
                A[3:,3:] = invSigma_Bj+I*(1-alpha)**2 * invsgma2
                
                # Initializing the B vectors
                b = np.zeros((6,1))
                b[:3] = np.reshape(invSigma_Fi @ mu_Fi + C*(alpha) * invsgma2,(3,1))
                b[3:] = np.reshape(invSigma_Bj @ mu_Bj + C*(1-alpha) * invsgma2,(3,1))

                # Solving for X
                
                X = np.linalg.solve(A, b)
                
                # Store the calculated values into F and B vectors
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                
                # Solving for alpha 
                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]
                
                # Calculating Likelihood for alphas
                like_C = - np.sum((np.atleast_2d(C).T -alpha*F-(1-alpha)*B)**2) * invsgma2
                
                # Calculating Foreground likelihood for foreground and background
                like_fg = (- ((F- np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F-np.atleast_2d(mu_Fi).T))/2)[0,0]
                like_bg = (- ((B- np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B-np.atleast_2d(mu_Bj).T))/2)[0,0]
                
                # Calculating the likelihood as a sum
                like = (like_C + like_fg + like_bg)

                # Setting condition to check maximum likelihood
                if like > maxlike:
                    a_best = alpha
                    maxlike = like
                    fg_best = F.ravel()
                    bg_best = B.ravel()

                # Setting condition for exiting loop execution
                if myiter >= maxIter or abs(like-lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1
                
                # Returning optimal foreground, background and alpha values
    return fg_best, bg_best, a_best


#computeMSE
# def getMSE(alpha_val, gt_img):
#     gt_img = np.double(gt_img)
#     gt_img = gt_img[:, :, 0]
#     mse_val = mean_squared_error(alpha_val, gt_img)
#     return mse_val

#computeSAD
def getSAD(alpha_val, gt_img):
    gt_img = np.double(gt_img)
    gt_img = gt_img[:, :, 0]
    sad_val = np.sum(np.abs(alpha_val - gt_img))
    return sad_val


def load_images(folder_path, len_seq = 10):
    image_files = os.listdir(folder_path)
    images = []
    images_name = []
    trimaps = []
    count = 0
    for file in image_files:
        if count >= len_seq:
            break
        file_path = os.path.join(folder_path, file)

        trimaps_path = os.path.join("Python", "Images","trimap_training_lowres","Trimap2", "{}".format(file))
        if os.path.isfile(file_path):
            try:
                img = np.array(Image.open(file_path))
                trimap = np.array(Image.open(trimaps_path))
                images.append(img)
                trimaps.append(trimap)
                images_name.append(file)
                count += 1
                print(f"Loaded image: {file}")
            except IOError:
                print(f"Could not open file as an image: {file}")
    return images, trimaps, images_name

#compute execution time of the first 10 pictures
def execution_time(images,trimaps, images_name):
    for i in range(len(images)):
        start_time = time.time()
        getBayesianMatte(images[i], trimaps[i], images_name[i])
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for image {i+1}: {execution_time:.5f} seconds")
        
def execution_longSeq(seqPath = "Python/Images/input_training_lowres", len_seq = 10):
    images, trimaps, images_name = load_images(seqPath,len_seq)
    execution_time(images, trimaps, images_name)


#composite image       
def composite(alpha, foreground, background):
    # Convert the alpha matte to a 3-channel grayscale image
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # Compute the composite image
    composite = alpha * foreground + (1 - alpha) * background

    # Clip the pixel values to the range [0, 1]
    composite = np.clip(composite, 0, 1)

    # Display the composite image
    plt.imshow(composite)
    plt.axis('off')
    plt.show()

    return composite

'''
Main Function
'''
def main(img_name = "GT01", user_selecr = False, save = False, show_FG = False, compositing = False):
    # Creating image object
    img_obj = initializeVariables()
    
    # Creating path
    if not user_selecr:
        IMG_PATH = os.path.join("Images","input_training_lowres","{}.png".format(img_name))
        TRIMAP_PATH = Image.open(os.path.join("Images","trimap_training_lowres", "Trimap2", "{}.png".format(img_name)))
    else:
        IMG_PATH = filedialog.askopenfilename(filetypes=[("Source Image Selection", "*.jpg;*.png;*.bmp")])
        TRIMAP_PATH = filedialog.askopenfilename(filetypes=[("Trimap Selection", "*.jpg;*.png;*.bmp")])

        TRIMAP_PATH = Image.open(TRIMAP_PATH)
        
    image = np.array(Image.open(IMG_PATH))
    
    # Creating trimap path
    image_trimap = np.array(ImageOps.grayscale(TRIMAP_PATH))
    
    # Calculating alpha matte
    #for i in tqdm(range(101), desc="Generating Matte", ascii=False, ncols=75):
    alpha = getBayesianMatte(image, image_trimap,  img_name, img_obj.N, img_obj.sigma, img_obj.min_N) 

    # Displaying alpha matte
    displayImage('Alpha Matte', alpha)

    if save:
        base_dir = 'Python/Result'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        # find the last subdirectory with a name that matches the pattern "exp<suffix>"
        sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        suffixes = [int(d.split('exp')[-1]) for d in sub_dirs if d.startswith('exp') and d.split('exp')[-1].isdigit()]
        if suffixes:
            next_suffix = max(suffixes) + 1
        else:
            next_suffix = 1
        
        # create the new subdirectory with the next suffix
        sub_dir = 'exp{}'.format(next_suffix)
        sub_dir_path = os.path.join(base_dir, sub_dir)
        os.makedirs(sub_dir_path)
        alpha = (alpha * 255).astype('uint8')
        # create a PIL Image object from the NumPy array
        pil_image = Image.fromarray(alpha)

        # save the PIL Image object as a PNG file in the subdirectory
        pil_image.save(os.path.join(sub_dir_path, 'alpha.png'))

        if show_FG:
            alpha_image = Image.open(os.path.join(sub_dir_path, 'alpha.png'))
            alpha_array = np.array(alpha_image.convert('L'))

            # create a new NumPy array for the foreground image by copying the original image and applying the alpha mask
            foreground_array = np.zeros_like(np.array(image))
            foreground_array[..., 0] = (image)[:,:,0] * (alpha_array / 255.0)
            foreground_array[..., 1] = (image)[:,:,1] * (alpha_array / 255.0)
            foreground_array[..., 2] = (image)[:,:,2] * (alpha_array / 255.0)

            # create a PIL Image object from the foreground NumPy array
            foreground_image = Image.fromarray(np.uint8(foreground_array))
            foreground_image.save(os.path.join(sub_dir_path,'foreground.png'))
            plt.imshow(foreground_image)
            plt.show()

            if compositing:
                background_path = filedialog.askopenfilename(filetypes=[("Background Selection", "*.jpg;*.png;*.bmp")])
                background_image = Image.open(background_path)
                
                background_array = np.array(background_image)

                # resize the background array to the same shape as the foreground array
                resized_background_array = np.array(Image.fromarray(np.uint8(background_image)).resize(foreground_array.shape[:2][::-1], resample=Image.Resampling.BILINEAR))


                # create a new NumPy array for the composite image by blending the foreground and background images using the alpha mask
                composition = np.zeros_like(foreground_array)
                composition[..., 0] = resized_background_array[:,:,0] * (1.0 - alpha_array / 255.0) + foreground_array[:,:,0]
                composition[..., 1] = resized_background_array[:,:,1] * (1.0 - alpha_array / 255.0) + foreground_array[:,:,1]
                composition[..., 2] = resized_background_array[:,:,2] * (1.0 - alpha_array / 255.0) + foreground_array[:,:,2]

                # create a PIL Image object from the composite NumPy array
                composite_image = Image.fromarray(np.uint8(composition))
                composite_image.save(os.path.join(sub_dir_path,'composition.png'))
                plt.imshow(composition)
                plt.show()






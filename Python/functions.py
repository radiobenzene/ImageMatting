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

"""
Function to display Image
Params:
    title: Image Title
    img: Display Image
Returns:
    Image box with a title 
"""
def displayImage(title, img):
    plt.show(img)
    plt.title(title)
    plt.show()

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
    image_trimap = np.array(ImageOps.grayscale(Image.open(os.path.join("Images","trimap_training_lowres","Trimap2", "{}.png".format(name)))))

    
    # Converting image to float type
    img = convertImage(img)
    trimap = convertImage(trimap)
    
    # Getting dimensions of images
    h,w,c = img.shape
    
    # Initializing Gaussian weighting
    gaussian_weights = fspecial((N,N),variance)
    gaussian_weights /= np.max(gaussian_weights)

    # We seperate the foreground specified in the trimap from the main image.
    fg_map = trimap == 1
    fg_actual = np.zeros((h,w,c))
    fg_actual = img * np.reshape(fg_map,(h,w,1))

    # We seperate the background specified in the trimap from the main image. 
    bg_map = trimap == 0
    bg_actual = np.zeros((h,w,c))
    bg_actual = img * np.reshape(bg_map,(h,w,1))
    
    # Creating empty alpha channel to fill in by the program
    unknown_map = np.logical_or(fg_map,bg_map) == False
    a_channel = np.zeros(unknown_map.shape)
    a_channel[fg_map] = 1
    a_channel[unknown_map] = np.nan

    # Finding total number of unkown pixels to be calculated
    n_unknown = np.sum(unknown_map)

    # Making the datastructure for finding pixel values and saving id they have been solved yet or not.
    A,B = np.where(unknown_map == True)
    not_visited = np.vstack((A,B,np.zeros(A.shape))).T

    print("Solving Image with {} unsovled pixels... Please wait...".format(len))

    # running till all the pixels are solved.
    while(sum(not_visited[:,2]) != n_unknown):
        last_n = sum(not_visited[:,2])

        # iterating for all pixels
        for i in range(n_unknown): 
            # checking if solved or not
            if not_visited[i,2] == 1:
                continue
            
            # If not solved, we try to solve
            else:
                # We get the location of the unsolved pixel
                y,x = map(int,not_visited[i,:2])
                
                # Creating an window which states what pixels around it are solved(forground/background)
                a_window = runWindow(a_channel[:, :, np.newaxis], x, y, N)[:,:,0]
                
                # Creating a window and weights of solved foreground window
                fg_window = runWindow(fg_actual,x,y,N)
                fg_weights = np.reshape(a_window**2 * gaussian_weights,-1)
                values_to_keep = np.nan_to_num(fg_weights) > 0
                fg_pixels = np.reshape(fg_window,(-1,3))[values_to_keep,:]
                fg_weights = fg_weights[values_to_keep]
        
                # Creating a window and weights of solved background window
                bg_window = runWindow(bg_actual,x,y,N)
                bg_weights = np.reshape((1-a_window)**2 * gaussian_weights,-1)
                values_to_keep = np.nan_to_num(bg_weights) > 0
                bg_pixels = np.reshape(bg_window,(-1,3))[values_to_keep,:]
                bg_weights = bg_weights[values_to_keep]
                
                # We come back to this pixel later if it doesnt has enough solved pixels around it.
                if len(bg_weights) < min_N or len(fg_weights) < min_N:
                    continue
                
                # If enough pixels, we cluster these pixels to get clustered colour centers and their covariance    matrices
                mean_fg, cov_fg = clusterElements(fg_pixels,fg_weights)
                mean_bg, cov_bg = clusterElements(bg_pixels,bg_weights)
                alpha_init = np.nanmean(a_window.ravel())
                
                # We try to solve our 3 equation 7 variable problem with minimum likelihood estimation
                fg_pred,bg_pred,alpha_pred = solve(mean_fg,cov_fg,mean_bg,cov_bg,img[y,x],0.7,alpha_init)

                # storing the predicted values in appropriate windows for use for later pixels.
                fg_actual[y, x] = fg_pred.ravel()
                bg_actual[y, x] = bg_pred.ravel()
                a_channel[y, x] = alpha_pred
                not_visited[i,2] = 1
                if(np.sum(not_visited[:,2])%1000 == 0):
                    print("Solved {} out of {}.".format(np.sum(not_visited[:,2]),len(not_visited)))

        if sum(not_visited[:,2]) == last_n:
            N += 2
            variance += 1 
            gaussian_weights = fspecial((N,N),variance)
            gaussian_weights /= np.max(gaussian_weights)
            print(N)

    return a_channel,n_unknown

 
"""
Function to display image
"""
def displayImage(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


# S is measurements vector - dim nxd
# w is weights vector - dim n
def clustFunc(S, w, minVar=0.05):
    mu, sigma = [], []
    nodes = []
    nodes.append(Node(S, w))

    while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        mu.append(node.mu)
        sigma.append(node.cov)

    return np.array(mu), np.array(sigma)


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
    
# Defining class to initialize variables
class initializeVariables:
    # Initialize Window Size
    N = 25 #120
    
    # Initialize Variance for Gaussian weighting
    sigma = 8; ###0.5
    
    # Initialize camera variance
    cam_sigma = 0.05
    
    # Initialize Minimum window size
    min_N = 10
    
    # Initialize Clustering variance
    clustering_variance = 0.05
    
    """ 
        Function to calculate MSE
        @FerniA
    """
#def getMSE(alpha_val, gt_img):
#    gt_img = np.double(gt_img)
#    gt_img = gt_img[:, :, 0]
#    mse_val = compare_mse(alpha_val, gt_img)
#    return mse_val


## To solve individual pixels
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_init, maxIter = 50, minLike = 1e-6):
    """
    mu_F - Mean of foreground pixel
    Sigma_F - Covariance Mat of foreground pixel
    mu_B, Sigma_B - Mean and Covariance of background pixel
    C, Sigma_C - Current pixel, and its variance
    alpha_init - starting alpha value
    maxIter - Iterations to solve the value of the pixel
    minLike - min likelihood to reach to stop before maxIterations. 
    """

    # Initializing Matrices
    I = np.eye(3)
    fg_best = np.zeros(3)
    bg_best = np.zeros(3)
    a_best = np.zeros(1)
    maxlike = -np.inf
    
    invsgma2 = 1/Sigma_C**2
    
    for i in range(mu_F.shape[0]):
        # Mean of Foreground pixel can have multiple possible values, iterating for all.
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])

        for j in range(mu_B.shape[0]):
            # Similarly, multiple mean values be possible for background pixel.
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            myiter = 1
            lastLike = -1.7977e+308

            # Solving Minimum likelihood through numerical methods
            while True:
                # Making Equations for AX = b, where we solve for X.abs
                # X here has 3 values of forground pixel (RGB) and 3 values for background
                A = np.zeros((6,6))
                A[:3,:3] = invSigma_Fi + I*alpha**2 * invsgma2
                A[:3,3:] = A[3:,:3] = I*alpha*(1-alpha) * invsgma2
                A[3:,3:] = invSigma_Bj+I*(1-alpha)**2 * invsgma2
                
                b = np.zeros((6,1))
                b[:3] = np.reshape(invSigma_Fi @ mu_Fi + C*(alpha) * invsgma2,(3,1))
                b[3:] = np.reshape(invSigma_Bj @ mu_Bj + C*(1-alpha) * invsgma2,(3,1))

                # Solving for X and storing values for Forground and Background Pixels 
                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                
                # Solving for value of alpha once F and B are calculated
                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]
                
                # Calculating likelihood value for
                like_C = - np.sum((np.atleast_2d(C).T -alpha*F-(1-alpha)*B)**2) * invsgma2
                like_fg = (- ((F- np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F-np.atleast_2d(mu_Fi).T))/2)[0,0]
                like_bg = (- ((B- np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B-np.atleast_2d(mu_Bj).T))/2)[0,0]
                like = (like_C + like_fg + like_bg)

                if like > maxlike:
                    a_best = alpha
                    maxlike = like
                    fg_best = F.ravel()
                    bg_best = B.ravel()

                if myiter >= maxIter or abs(like-lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1
    return fg_best, bg_best, a_best
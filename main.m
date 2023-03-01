% Bayesian Matting with Orchard-Boumann Clustering
% Group - Atomic Reactors

% Reading image
img = imread("Images\input-small.png");

% Reading trimap 
trimap = imread("Images\trimap.png");

% Converting both image files to double
img = im2double(img);
trimap = im2double(trimap);

img_obj = initializeVariable();
% Starting timer here
tic;
% Performing Bayesian Matting here
[Fground, Bground, alpha_val] = getBayesianMatte(img, trimap, img_obj);

% Ending timer here
toc

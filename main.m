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

% Generate composite image here
composite_img = generateCompositeImage(Fground, Bground, alpha_val);
% Ending timer here
toc;

% Displaying Original Image
figure(1);
imshow("Images\input-small.png");
title('Original Image');
 
% Displaying Alpha Matte
figure(2);
imshow(alpha_val);
title('Alpha Matte');
shg; hold on;

figure(3);
imshow(composite_img);
title('Composite Image');





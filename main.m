% Bayesian Matting with Orchard-Boumann Clustering
% Group - Atomic Reactors

% Reading image
img = imread("Images\bear\input.jpg");

% Reading trimap 
trimap = imread("Images\bear\trimap.png");

% Converting both image files to double
img = im2double(img);
trimap = im2double(trimap);

img_obj = initializeVariable();
% Starting timer here
tic;
% Performing Bayesian Matting here
[Fground, Bground, alpha_val] = getBayesianMatte(img, trimap, img_obj);
% Generate composite image here
newB = imread("Images\new_background.jfif");
newB = im2double(newB);
newB = imresize(newB, [282 420]);
composite_img = generateCompositeImage(Fground, newB, alpha_val);
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





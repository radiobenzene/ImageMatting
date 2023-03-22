% Load the input image and trimap
img = imread('Images\input-small.png');
trimap = imread('Images\trimap.png');

% Convert the image and trimap to double precision and normalize
img = double(img);
trimap = double(trimap);
% trimap = trimap ./ 255;

% Get the size of the image
[m, n, c] = size(img);

% Calculate the foreground, background, and unknown pixels
fg = trimap > 0.99;
bg = trimap < 0.01;
unk = ~(fg | bg);

% Create the Laplacian matrix
Laplacian = del2(img);

% Calculate the alpha matte
alpha = zeros(m, n);
for i = 1:c
    alpha(unk(:, :, 1)) = alpha(unk(:, :, 1)) + Laplacian(unk(:, :, 1), i) .^ 2;
end
alpha = 1 - sqrt(alpha ./ c);
alpha(bg(:, :, 1)) = 0;
alpha(fg(:, :, 1)) = 1;

% Save the output alpha matte
imwrite(alpha, 'output_alpha.png');
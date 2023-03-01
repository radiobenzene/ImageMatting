% Defining a class which stores all values
classdef initializeVariable
    properties
        N = 20; % Windowing size
        sigma = 50; % Variance for Gaussian
        cam_sigma = 0.05; % Camera Variance
        min_N = 10; % Minimum Window size
    end
end

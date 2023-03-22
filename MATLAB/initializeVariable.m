% Defining a class which stores all values
classdef initializeVariable
    properties
        N = 120; % Windowing size
        sigma = 50; % Variance for Gaussian
        cam_sigma = 0.05; % Camera Variance
        min_N = 10; % Minimum Window size
        clustering_variance = 0.90;
    end
end

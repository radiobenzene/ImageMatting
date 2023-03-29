classdef BayesianMatting_unittest < matlab.unittest.TestCase
    methods (Test)
        %% Test1: check image loaded is not Empty
        function testImageifEmpty(testCase)
            sourceImg = imread("Images\imagefortesting\input_training_lowres\GT01.png");
            trimapImg = imread("Images\imagefortesting\trimap_training_lowres\Trimap1\GT01.png");
            GTImg = imread("Images\imagefortesting\gt_training_lowres\GT01.png");

            verifyNotEmpty(testCase,sourceImg, "Check the source image loading, now it is empty");
            verifyNotEmpty(testCase,trimapImg, "Check the trimap loading, now it is empty");
            verifyNotEmpty(testCase,GTImg, "Check the Ground Truth image loading, now it is empty");
        end

        %% Test2: Image size loading check
        function testInputImageSize(testCase)
            sourceImg = imread("Images\imagefortesting\input_training_lowres\GT01.png");
            trimapImg = imread("Images\imagefortesting\trimap_training_lowres\Trimap1\GT01.png");
            GTImg = imread("Images\imagefortesting\gt_training_lowres\GT01.png");
            
            expect_src = rand(497,800,3);
            expect_trimap = rand(497,800);
            expect_GT = rand(497,800,3);

            verifySize(testCase,(sourceImg),size(expect_src));
            verifySize(testCase,(trimapImg), size(expect_trimap));
            verifySize(testCase,(GTImg), size(expect_GT));
        end

        %% Test 3: Test Image format
        function testImageFormat(testCase)
            sourceImg = imread("Images\imagefortesting\input_training_lowres\GT01.png");
            trimapImg = imread("Images\imagefortesting\trimap_training_lowres\Trimap1\GT01.png");
            GTImg = imread("Images\imagefortesting\gt_training_lowres\GT01.png");
            
            verifyClass(testCase, sourceImg, "uint8");
            verifyClass(testCase, trimapImg, "uint8");
            verifyClass(testCase, GTImg, "uint8");
            
            % the double format is used for calcaluating F,B and Alpha
            sourceImg = im2double(sourceImg);
            trimapImg = im2double(trimapImg);
            GTImg = im2double(GTImg);

            verifyClass(testCase, sourceImg, "double");
            verifyClass(testCase, trimapImg, "double");
            verifyClass(testCase, GTImg, "double");
        end       

        %% Test 4: testOutputSize
        function testOutputSize(testCase)
            img = zeros(20, 20, 3);
            trimap = ones(20, 20);
            c_obj = initializeVariable();
            [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);
            testCase.verifyEqual(size(F), [20, 20, 3]);
            testCase.verifyEqual(size(B), [20, 20, 3]);
            testCase.verifyEqual(size(alpha_channel), [20, 20]);
        end
        
        %% Test 5: If the Trimap is characterize as fully Background
        function testAllBackground(testCase)
            img = rand(5, 5, 3);
            trimap = zeros(5, 5);

            c_obj = initializeVariable();

            [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);

            expected_F = zeros(size(img));
            expected_alpha = zeros(size(trimap));

            testCase.verifyEqual(F, expected_F);
            testCase.verifyEqual(B, img);
            testCase.verifyEqual(alpha_channel, expected_alpha);
        end
        
        %% Test 6: If the Trimap is characterize as fully Foreground
        function testAllForeground(testCase)
            img = rand(5, 5, 3);
            trimap = ones(5, 5);

            c_obj = initializeVariable();

            [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);

            expected_B = zeros(size(img));
            expected_alpha = ones(size(trimap));

            testCase.verifyEqual(F, img);
            testCase.verifyEqual(B, expected_B);
            testCase.verifyEqual(alpha_channel, expected_alpha);
        end
        
        %% Test 7: Test trimap segmentation
        function testTrimapSegmentation(testCase)
            sourceImg = imread("Images\imagefortesting\input_training_lowres\GT01.png");
            trimapImg = imread("Images\imagefortesting\trimap_training_lowres\Trimap1\GT01.png");
            GTImg = imread("Images\imagefortesting\gt_training_lowres\GT01.png");
    
            img = im2double(sourceImg);
            trimap = im2double(trimapImg);
            gt_image = im2double(GTImg);

            background_mask = (trimap==0); % Background where trimap values = 0
            foreground_mask = (trimap==1); % Foreground where trimap values = 1
            unknown_area_mask= ~background_mask&~foreground_mask; % If neither, then unknown
            
            verifySize(testCase, (background_mask), size(trimap));
            verifySize(testCase, (foreground_mask), size(trimap));
            verifySize(testCase, (unknown_area_mask), size(trimap));
        end

        %% Test 8: Test Foreground and background size
        function testFBsize(testCase)
            sourceImg = imread("Images\imagefortesting\input_training_lowres\GT01.png");
            trimapImg = imread("Images\imagefortesting\trimap_training_lowres\Trimap1\GT01.png");
            GTImg = imread("Images\imagefortesting\gt_training_lowres\GT01.png");
    
            img = im2double(sourceImg);
            trimap = im2double(trimapImg);
            gt_image = im2double(GTImg);
            background_mask = (trimap==0); % Background where trimap values = 0
            foreground_mask = (trimap==1); % Foreground where trimap values = 1
            unknown_area_mask= ~background_mask&~foreground_mask;
            F = img; 
            F(repmat(~foreground_mask,[1,1,3])) = 0;
            B=img; 
            B(repmat(~background_mask,[1,1,3])) = 0;
            alpha_channel = zeros(size(trimap));
            alpha_channel(foreground_mask) = 1;
            alpha_channel(unknown_area_mask) = NaN;

            verifySize(testCase, F, size(img));
            verifySize(testCase, B, size(img));
            verifySize(testCase, alpha_channel, size(trimap));
        end

        %% Test 9: Test Gaussian Weight
        function testGaussianWeight(testCase)
            c_obj = initializeVariable();
            n = c_obj.N; 
            sigma = c_obj.sigma;

            gaussian_weighting = fspecial('gaussian', n, sigma); 

            % initialize filter kernel
            h = zeros(n, n);
            
            % calculate filter coefficients using Gaussian distribution formula
            for i = 1:n
                for j = 1:n
                    x = i - (n+1)/2;
                    y = j - (n+1)/2;
                    h(i, j) = exp(-(x^2 + y^2) / (2*sigma^2)) / (2*pi*sigma^2);
                end
            end
            h = h / sum(h(:));
            % The gaussian should return a NxN windows
            verifySize(testCase,(gaussian_weighting),[c_obj.N,c_obj.N])
            verifyEqual(testCase,gaussian_weighting, h, 'RelTol', 1e-6);
        end
        
        %% Test 10: Test unknown reigion shrinking the boundaries of objects in the image.
        function testUnkownShrink(testCase)
            % Create a binary image
            I = [0 0 0 0 0 0 0;
                 0 1 1 1 0 0 0;
                 0 1 1 1 0 0 0;
                 0 1 1 1 0 0 0;
                 0 0 0 0 0 0 0];
             
            % Create a 3x3 square structuring element
            SE = strel('square', 3);
            
            % Perform erosion on the image using the structuring element
            J = imerode(I, SE);
            
            expect_J = [0	0	0	0	0	0	0;
                        0	0	0	0	0	0	0;
                        0	0	1	0	0	0	0;
                        0	0	0	0	0	0	0;
                        0	0	0	0	0	0	0];
            verifyEqual(testCase, J, expect_J)
        end
        
        %% Test case 11: square window in center of image
        function testRunWindowCenter(testCase)
            img_area = ones(10, 10, 3);
            x = 5;
            y = 5;
            N = 3;
            expected_window = ones(3, 3, 3);
            window_val = runWindow(img_area, x, y, N);
            verifyEqual(testCase, window_val, expected_window);
        end

        %% Test case 12: rectangular window at top left of image
        function testRunWindowsTF(testCase)
            img_area = ones(10, 10, 3);
            x = 3;
            y = 3;
            N = 4;
            expected_window = ones(4, 4, 3);
            window_val = runWindow(img_area, x, y, N);
            verifyEqual(testCase, window_val, expected_window);
        end

        %% Test case 13: rectangular window at bottom right of image
        function testRunwindowsBR(testCase)
            img_area = ones(10, 10, 3);
            x = 9;
            y = 9;
            N = 4;
            expected_window = ones(4, 4, 3);
            window_val = runWindow(img_area, x, y, N);
            verifyEqual(testCase, window_val, expected_window);
        end

        %% Test case 14: window partially outside image
        function testWindowOutside(testCase)
            img_area = ones(10, 10, 3);
            x = 9;
            y = 9;
            N = 4;
            expected_window = ones(3, 3, 3);
            window_val = runWindow(img_area, x, y, N);
            verifyEqual(testCase, window_val(2:4, 2:4, :), expected_window);
        end

        %% Test Case 15: Test ClusteringOutput
        function testClustering(testCase)
            c_obj = initializeVariable();
            load TestCase\foreground_pixels.mat
            load TestCase\foreground_weights.mat 
            
            [foregound_mean,foreground_cov] = ...
                orchardBoumannClustering( ...
                foreground_pixels, ...
                foreground_weights, ...
                c_obj.clustering_variance);
            foregound_mean_size = ones(3,1);
            foreground_cov_size = ones(3,3);
            verifyEqual(testCase, size(foregound_mean),size(foregound_mean_size));
            verifyEqual(testCase, size(foreground_cov),size(foreground_cov_size));
        end
        
        %% Test Case 16: Test Likelihood
        function testLikelihood(testCase)
            foreground_mean = [0.8116; 0.3448; 0.608];
            foreground_cov = [0.0025,0,0;
                             0,0.0025,0;
                             0,0,0.0025];
            background_mean = [0.3024;0.3431;0.2598];
            background_cov = [0.0025,0,0;
                             0,0.0025,0;
                             0,0,0.0025];
            c = [0.2039;0.2824;0.1641];
            cameraSigma = 0.0500;
            alpha_init = 0.5683;
            max_iterations = 100;
            min_likelihood = 1.0000e-6;

            [fg_val,bg_val,a_values]= likelihoodSolver( ...
                foreground_mean, ...
                foreground_cov, ...
                background_mean, ...
                background_cov, ...
                c, ...
                cameraSigma, ...
                alpha_init, ...
                max_iterations, ...
                min_likelihood);

            verifyEqual(testCase, size(fg_val), size(foreground_mean));
            verifyEqual(testCase, size(bg_val), size(background_mean));
        end
    end
end

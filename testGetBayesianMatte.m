function tests = testGetBayesianMatte
    tests = functiontests(localfunctions);
end

function testOutputSize(testCase)
    % Test that the output sizes of F, B, and alpha_channel are correct
    img = zeros(20, 20, 3); 
    trimap = ones(20, 20); 
    c_obj.N = 120;
    c_obj.sigma = 50;
    c_obj.min_N = 10;
    c_obj.clustering_variance = 0.90;
    c_obj.cam_sigma = 0.05;

    [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);
    verifyEqual(testCase, size(F), [20, 20, 3]);
    verifyEqual(testCase, size(B), [20, 20, 3]);
    verifyEqual(testCase, size(alpha_channel), [20, 20]);
end

function testAlphaValues(testCase)
    % Test that the alpha values are between 0 and 1
    img = zeros(20, 20, 3); 
    trimap = ones(20, 20); 
    c_obj.N = 120;
    c_obj.sigma = 50;
    c_obj.min_N = 10;
    c_obj.clustering_variance = 0.90;
    c_obj.cam_sigma = 0.05;

    [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);
    % check if the alpha lie between 0 - 1
    verifyTrue(testCase, all(alpha_channel(:) >= 0 & alpha_channel(:) <= 1)); 
end

function testForeground(testCase)
    % checks whether all the alpha values are equal to 1
    img = zeros(20, 20, 3); 
    trimap = ones(20, 20); 
    c_obj.N = 120;
    c_obj.sigma = 50;
    c_obj.min_N = 10;
    c_obj.clustering_variance = 0.90;
    c_obj.cam_sigma = 0.05;

    [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);
    verifyTrue(testCase, all(alpha_channel(trimap == 1) == 1));
end

function testBackground(testCase)
    % checks whether all the alpha values are equal to 0
    img = zeros(20, 20, 3); 
    trimap = zeros(20, 20); 
    c_obj.N = 120;
    c_obj.sigma = 50;
    c_obj.min_N = 10;
    c_obj.clustering_variance = 0.90;
    c_obj.cam_sigma = 0.05;

    [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj);
    verifyTrue(testCase, all(alpha_channel(trimap == 0) == 0));
end

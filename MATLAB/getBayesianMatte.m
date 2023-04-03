function [F, B, alpha_channel] = getBayesianMatte(img, trimap, c_obj)

    % Setting masks for foreground, background and unknown
    background_mask = (trimap==0); % Background where trimap values = 0
    foreground_mask = (trimap==1); % Foreground where trimap values = 1
    unknown_area_mask= ~background_mask&~foreground_mask; % If neither, then unknown

    % initialize Foreground values
    F = img; 
    F(repmat(~foreground_mask,[1,1,3])) = 0;

    % Initialize Background values
    B=img; 
    B(repmat(~background_mask,[1,1,3])) = 0;

    % Initialize alpha values
    alpha_channel = zeros(size(trimap));
    alpha_channel(foreground_mask) = 1;
    alpha_channel(unknown_area_mask) = NaN;

    % Initializing total unknown points
    unknown_points=sum(unknown_area_mask(:));

    % Gaussian Weighting
    gaussian_weighting = fspecial('gaussian', c_obj.N, c_obj.sigma); 

    % Normalizing weighting
    gaussian_weighting = gaussian_weighting/max(gaussian_weighting(:));

    % square structuring element for eroding the unknown region(s)
    se=strel('square',3);
    
    % Setting initial interation
    n=1;
    unknown_region=unknown_area_mask;
    iter = 1;

    while ((n < unknown_points) && (iter < 100)) % - 1000
        % Getting a block of 3 * 3 which is unknown region
        unknown_region=imerode(unknown_region,se);

        unknown_pixels=~unknown_region&unknown_area_mask;
        [Y,X]=find(unknown_pixels); 

        for i=1:length(Y)
            fprintf('Working on %d out of %d unknown points\n',n,unknown_points);
            %imshow(alpha_channel);

            % Getting current pixel
            x=X(i); y=Y(i);
            c = reshape(img(y,x,:),[3,1]);

            % take surrounding alpha values
            a_values = runWindow(alpha_channel,x,y,c_obj.N);

            % take surrounding foreground pixels
            foreground_pixels = runWindow(F, x, y, c_obj.N);
            foreground_weights = (a_values .^ 2) .* gaussian_weighting;
            foreground_pixels = reshape(foreground_pixels, c_obj.N * c_obj.N, 3);
            foreground_pixels = foreground_pixels(foreground_weights>0,:);
            foreground_weights = foreground_weights(foreground_weights>0);

            % take surrounding background pixels
            background_pixels = runWindow(B, x, y, c_obj.N);
            background_weights = ((1 - a_values) .^ 2) .* gaussian_weighting;
            background_pixels = reshape(background_pixels, c_obj.N * c_obj.N, 3);
            background_pixels = background_pixels(background_weights > 0, :);
            background_weights = background_weights(background_weights > 0);

            % if not enough data, return to it later...
            if ((length(foreground_weights) < c_obj.min_N) || (length(background_weights) < c_obj.min_N))
                continue;
            end

            % partition foreground and background pixels to clusters (in a
            % weighted manner)
            [foregound_mean,foreground_cov] = ...
                orchardBoumannClustering( ...
                foreground_pixels, ...
                foreground_weights, ...
                c_obj.clustering_variance);

            [background_mean,background_cov] = ...
                orchardBoumannClustering( ...
                background_pixels, ...
                background_weights, ...
                c_obj.clustering_variance);
            
            % Recaliberating variance with camera variance
            foreground_cov = recaliberateVariance(foreground_cov, c_obj.cam_sigma);
            background_cov = recaliberateVariance(background_cov, c_obj.cam_sigma);

            % Setting the initial alpha values as the mean of the pixels
            % around it
            alpha_init=mean(a_values(:), 'omitnan'); 
            
            % Setting maximum iterations
            max_iterations = 100;

            % Setting minimum likelihood
            min_likelihood = 1e-6;
            [fg_val,bg_val,a_values]= likelihoodSolver( ...
                foregound_mean, ...
                foreground_cov, ...
                background_mean, ...
                background_cov, ...
                c, ...
                c_obj.cam_sigma, ...
                alpha_init, ...
                max_iterations, ...
                min_likelihood);
            
            % Updating Foreground value
            F(y,x,:) = fg_val;

            % Updating Background value
            B(y,x,:) = bg_val;

            % Updating alpha values
            alpha_channel(y,x) = a_values;

            % Removing points from the unknown area
            unknown_area_mask(y,x) = 0; 

            % Increment counter
            n = n + 1;
        end
        iter = iter + 1;
    end

end
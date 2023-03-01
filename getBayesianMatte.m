function [F, B, alpha] = getBayesianMatte(img, trimap, c_obj)

    % Setting masks for foreground, background and unknown
    bgmask = (trimap==0); % Background where trimap values = 0
    fgmask = (trimap==1); % Foreground where trimap values = 1
    unkmask= ~bgmask&~fgmask; % If neither, then unknown

    % initialize Foreground values
    F = img; 
    F(repmat(~fgmask,[1,1,3])) = 0;

    % Initialize Background values
    B=img; 
    B(repmat(~bgmask,[1,1,3])) = 0;

    % Initialize alpha values
    alpha = zeros(size(trimap));
    alpha(fgmask) = 1;
    alpha(unkmask) = NaN;

    % Initializing total unknown points
    nUnknown=sum(unkmask(:));

    % Gaussian Weighting
    g = fspecial('gaussian', c_obj.N, c_obj.sigma); 
    g = g/max(g(:));

    % square structuring element for eroding the unknown region(s)
    se=strel('square',3);

    n=1;
    unkreg=unkmask;
    iter = 1;

    while ((n < nUnknown) && (iter < 100)) % - 1000
        unkreg=imerode(unkreg,se);
        unkpixels=~unkreg&unkmask;
        [Y,X]=find(unkpixels); 

        for i=1:length(Y)
            fprintf('processing %d/%d\n',n,nUnknown);

            % take current pixel
            x=X(i); y=Y(i);
            c = reshape(img(y,x,:),[3,1]);

            % take surrounding alpha values
            a = runWindow(alpha,x,y,c_obj.N);

            % take surrounding foreground pixels
            f_pixels=runWindow(F,x,y,c_obj.N);
            f_weights=(a.^2).*g;
            f_pixels=reshape(f_pixels,c_obj.N * c_obj.N,3);
            f_pixels = f_pixels(f_weights>0,:);
            f_weights = f_weights(f_weights>0);

            % take surrounding background pixels
            b_pixels=runWindow(B,x,y,c_obj.N);
            b_weights=((1-a).^2).*g;
            b_pixels=reshape(b_pixels,c_obj.N * c_obj.N,3);
            b_pixels=b_pixels(b_weights>0,:);
            b_weights=b_weights(b_weights>0);

            % if not enough data, return to it later...
            if length(f_weights)<c_obj.min_N || length(b_weights)<c_obj.min_N
                continue;
            end

            % partition foreground and background pixels to clusters (in a
            % weighted manner)
            [mu_f,Sigma_f]= orchardBoumannClustering(f_pixels, f_weights, c_obj.clustering_variance);
            [mu_b,Sigma_b]= orchardBoumannClustering(b_pixels, b_weights, c_obj.clustering_variance);
            
            % Recaliberating variance with camera variance
            Sigma_f=recaliberateVariance(Sigma_f, c_obj.cam_sigma);
            Sigma_b=recaliberateVariance(Sigma_b, c_obj.cam_sigma);

            % set initial alpha value to mean of surrounding pixels
            alpha_init=mean(a(:), 'omitnan'); %nanmean(a(:));
            
            % solve for current pixel
            max_iter = 100;
            min_likelihood = 1e-6; %1e-6;
            [f,b,a]=likelihoodSolver(mu_f,Sigma_f,mu_b,Sigma_b, c ,c_obj.cam_sigma, alpha_init, max_iter, min_likelihood);
            
            F(y,x,:)=f;
            B(y,x,:)=b;
            alpha(y,x)=a;
            unkmask(y,x)=0; % remove from unkowns
            n = n + 1;
        end
        iter = iter + 1;
    end

end
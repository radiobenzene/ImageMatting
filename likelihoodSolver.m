% Function to solve likelihood problem
% Params
%   Fground_mean - mean of foreground clusters
%   Fground_cov - sigma/covariance of foreground clusters
%   Bground_mean - mean of background clusters
%   Bground_cov - sigma/covariance of background clusters
%   pixel_val - current pixel value
%   pixel_sigma - covar of pixel
%   initial_alpha_value - pre-set alpha value
%   max_iterations - pre-set maximum iterations
%   min_likelihood - pre-set minimum likelihood
function [F,B,alpha_val]=likelihoodSolver(Fground_mean, ...
    Fground_cov, ...
    Bground_mean, ...
    Bground_cov, ...
    pixel_val, ...
    pixel_sigma, ...
    initial_alpha_val, ...
    max_iterations, ...
    min_likelihood)
  
I=eye(3);

% Initializing empty array that will have all values
arr=[];

% Looping thround the clustered values
for i = 1:size(Fground_mean, 2)

    % Getting Foreground mean value at i
    i_Fground_mean = Fground_mean(:, i);
    inverse_cov_i_Foreground = inv(Fground_cov(:,:,i));
            
    for j = 1:size(Bground_mean,2)

        i_Bground_mean = Bground_mean(:,j);
        inverse_cov_i_Background = inv(Bground_cov(:,:,j));
        
        alpha_val = initial_alpha_val;
        counter = 1;
        % Setting likelihood to be as -maximum_val
        lastLike=-realmax;

        while (true)
            
            % solve for F,B
            part_A = inverse_cov_i_Foreground + I * (alpha_val ^ 2 / pixel_sigma ^ 2);
            part_B = I * alpha_val * (1-alpha_val) / pixel_sigma ^ 2;
            part_A_1 = I*((alpha_val*(1-alpha_val))/pixel_sigma^2);
            part_B_1 = inverse_cov_i_Background+I*(1-alpha_val)^2/pixel_sigma^2;
            A=[part_A , part_B; 
                part_A_1, part_B_1];
             
            b=[inverse_cov_i_Foreground * i_Fground_mean+pixel_val * (alpha_val/pixel_sigma^2); 
               inverse_cov_i_Background * i_Bground_mean+pixel_val * ((1-alpha_val)/pixel_sigma^2)];
           
            % Solving for X
            solution_val=A\b;
            
            % Storing values in F
            F=max(0,min(1,solution_val(1:3)));

            % Storing values in B
            B=max(0,min(1,solution_val(4:6)));
            
            % solve for alpha
            alpha_val=max(0,min(1,((pixel_val-B)' * (F-B))/sum((F-B).^2)));
            
            % calculate likelihood
            L_C_val = -sum((pixel_val - alpha_val * F - (1 - alpha_val) * B) .^ 2) / ...
                pixel_sigma;
            L_F_val = -((F - i_Fground_mean)' * inverse_cov_i_Foreground * ...
                (F - i_Fground_mean))/2;
            L_B=-((B-i_Bground_mean)'*inverse_cov_i_Background* ...
                (B-i_Bground_mean))/2;

            % Adding all three together
            combined_likelihood= L_C_val + L_F_val + L_B;
            
            % If reached limit, then break
            if counter>=max_iterations || abs(combined_likelihood-lastLike)<=min_likelihood
                break;
            end
            
            lastLike=combined_likelihood;
            counter=counter+1;
        end
        
        % Updating our array structure to store values
        val.F=F;
        val.B=B;
        val.alpha=alpha_val;
        val.like=combined_likelihood;
        arr=[arr val];
    end
end

% Get maximum likelihood values
[~,ind]=max([arr.like]);

% Determine F value
F=arr(ind).F;

% Determine B value
B=arr(ind).B;

% Determine alpha value
alpha_val=arr(ind).alpha;


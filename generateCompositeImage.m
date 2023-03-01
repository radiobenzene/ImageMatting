% Function to build composite image
function c_image = generateCompositeImage(F_ground, B_ground, alpha_val)
    alpha_val=repmat(alpha_val,[1,1,size(F_ground,3)]);
    c_image = alpha_val .* F_ground + (1 - alpha_val) .* B_ground;
end

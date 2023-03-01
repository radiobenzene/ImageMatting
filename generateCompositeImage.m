% Function to build composite image
function c_image = generateCompositeImage(F_ground, B_ground, alpha_val)
    c_image = alpha_val .* F_ground + (1 - alpha_val) .* B_ground;
end

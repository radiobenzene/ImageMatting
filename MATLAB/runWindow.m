% Function to get values in a specific window of size N * N
function window_val = runWindow(img_area, x, y, N)
    % Getting dimensions of the area
    [height,width,channels]=size(img_area);

    % Setting windows half
    half_win_size = floor(N/2);
    N_1=half_win_size; 
    N_2=N-half_win_size-1;
    
    % Setting an initial value as NaN
    window_val=nan(N,N,channels);
    
    % Setting max and min values for x-axis
    x_min = max(1, x - N_1);
    x_max = min(width, x + N_2);
    
    % Setting max and min values for y-axis
    y_min = max(1, y - N_1);
    y_max = min(height, y + N_2);
    
    % Getting pixel values for x coordinates (min and max)
    pixel_x_min = half_win_size - (x - x_min) + 1; 
    pixel_x_max = half_win_size + (x_max - x) + 1;
    
    % Getting pixel values for y coordinates (min and max)
    pixel_y_min = half_win_size - (y - y_min) + 1; 
    pixel_y_max = half_win_size + (y_max - y) + 1;
    
    % Setting window values
    window_val(pixel_y_min:pixel_y_max, pixel_x_min:pixel_x_max, :)= ...
        img_area(y_min:y_max, x_min:x_max, :);
end
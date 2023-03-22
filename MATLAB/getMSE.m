function mse_val = getMSE(alpha_val, gt_img)
    gt_img = im2double(gt_img);
    gt_img = gt_img(:, :, 1);
    mse_val = immse(alpha_val, gt_img);
end

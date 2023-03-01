function Sigma = recaliberateVariance(Sigma,sigma_C)

    Sigma=zeros(size(Sigma));
    for i=1:size(Sigma,3)
        Sigma_i=Sigma(:,:,i);
        [U,S,V]=svd(Sigma_i);
        Sp=S+diag([sigma_C^2,sigma_C^2,sigma_C^2]);
        Sigma(:,:,i)=U*Sp*V';
    end
end
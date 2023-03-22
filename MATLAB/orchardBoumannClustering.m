% Function to perform Orchard Boumann Clustering
% Params:
%   vector_val - vector
%   weight_val - corresponding weights
%   variance_lim - minimum variance threshold
function [mean_val,cov_val]=orchardBoumannClustering( ...
    vector_val, ...
    weight_val, ...
    variance_lim)
    
    % Initializing one big cluster
    C1.X=vector_val;
    C1.w=weight_val;

    C1=getClusterStats(C1);

    nodes=C1;
    
    while (max([nodes.lambda])>variance_lim)
        nodes=SplitNodes(nodes);
    end
    
    for i=1:length(nodes)
        mean_val(:,i)=nodes(i).q;
        cov_val(:,:,i)=nodes(i).R;
    end
end
% Returns covariance and mean
function C=getClusterStats(C)
    
    accuracy = 1e-5;
    W=sum(C.w);
    % weighted mean
    C.q=sum(repmat(C.w,[1,size(C.X,2)]).*C.X,1)/W;

    % weighted covariance
    t = (C.X - repmat(C.q, [size(C.X,1), 1])) .* (repmat(sqrt(C.w), [1,size(C.X, 2)]));

    C.R = (t' * t) / W + accuracy * eye(3);
    
    C.wtse = sum(sum((C.X-repmat(C.q,[size(C.X,1),1])).^2));
    
    % Getting the eigenvectors and eigenvalues
    [V,D]=eig(C.R);

    % Getting the exact value in the third position of array
    C.e=V(:,3);
    C.lambda=D(9);

end
% Function to split eigenvalues
function nodes = SplitNodes(nodes)
    
    % Get maximum lambda value
    [x,i]=max([nodes.lambda]);
    Ci=nodes(i);
    
    % Get a logical index
    index_val = (Ci.X * Ci.e) <= (Ci.q * Ci.e);

    % Updating value for vector, weights where index_value = 1
    Ca.X=Ci.X(index_val,:); 
    Ca.w=Ci.w(index_val);

    % Updating value for vector, weights where index_value = 0
    Cb.X=Ci.X(~index_val,:); 
    Cb.w=Ci.w(~index_val);

    % Get cluster statistics
    Ca=getClusterStats(Ca);
    Cb=getClusterStats(Cb);

    nodes(i)=[];
    nodes=[nodes,Ca,Cb];
end


% Function to perform Orchard Boumann Clustering
% Params:
%   vector_val - vector
%   weight_val - corresponding weights
%   variance_lim - minimum variance threshold
function [mu,Sigma]=orchardBoumannClustering(vector_val,weight_val,variance_lim)

% Initializing one big cluster
C1.X=vector_val;
C1.w=weight_val;
C1=calc(C1);
nodes=C1;

while (max([nodes.lambda])>variance_lim)
    nodes=split(nodes);
end

for i=1:length(nodes)
    mu(:,i)=nodes(i).q;
    Sigma(:,:,i)=nodes(i).R;
end

% calculates cluster statistics
function C=calc(C)

W=sum(C.w);
% weighted mean
C.q=sum(repmat(C.w,[1,size(C.X,2)]).*C.X,1)/W;
% weighted covariance
t=(C.X-repmat(C.q,[size(C.X,1),1])).*(repmat(sqrt(C.w),[1,size(C.X,2)]));
C.R=(t'*t)/W + 1e-5*eye(3);

C.wtse = sum(sum((C.X-repmat(C.q,[size(C.X,1),1])).^2));

[V,D]=eig(C.R);
C.e=V(:,3);
C.lambda=D(9);

% splits maximal eigenvalue node in direction of maximal variance
function nodes=split(nodes)

[x,i]=max([nodes.lambda]);
Ci=nodes(i);
idx=Ci.X*Ci.e<=Ci.q*Ci.e;
Ca.X=Ci.X(idx,:); Ca.w=Ci.w(idx);
Cb.X=Ci.X(~idx,:); Cb.w=Ci.w(~idx);
Ca=calc(Ca);
Cb=calc(Cb);
nodes(i)=[]; % remove the i'th nodes and replace it with its children
nodes=[nodes,Ca,Cb];


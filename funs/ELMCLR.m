function [X, W, y, A, evs, error] = ELMCLR(X, c, d, k, r, islocal, num_hidden_neurons,alpha)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% d: projected dimension
% k: number of neighbors to determine the initial graph, and the parameter r if r<=0
% r: paremeter, which could be set bo a large enough value. If r<0, then it is determined by algorithm with k
% islocal:
%           1: only update the similarities of the k neighbor pairs, the neighbor pairs are determined by the distances in the original space
%           0: update all the similarities
% W: dim*d projection matrix
% y: num*1 cluster indicator vector
% A: num*num learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations


% Parameters
opts.tol = 1e-10;
opts.issym=1;
opts.disp = 0;


NITER = 30;
num = size(X,2);

if nargin < 6
    islocal = 1;
end

if nargin < 5
    r = -1;
end
if nargin < 4
    k = 15;
end
if nargin < 3
    d = c-1;
end

%%% Initialize graph
distx_x = L2_distance_1(X,X);
distx_x_norm_v = sqrt(sum(abs(distx_x).^2,2));
distx_x_norm_m = sqrt(distx_x_norm_v*distx_x_norm_v');
distx_x_norm = distx_x./distx_x_norm_m;
distX = distx_x_norm;

[distX1, idx] = sort(distX,2);
A = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

if r <= 0
    r = mean(rr);
end
lambda = r;


A = (A+A')/2;

%%% Initialize ELM random neurons

input_weight = rand(size(X,1), num_hidden_neurons)*2-1;
bias = rand(1,num_hidden_neurons);
tempH = X' * input_weight;
tempH = bsxfun(@plus,tempH,bias);
H = 1 ./ (1 + exp(-tempH));
clear tempH;
X = H';


for iter = 1:NITER
    
    A_old = A;
    D = diag(sum(A));
    L = D-A;
    
    % Update embedding

    if num_hidden_neurons<num
        X_center = X-repmat(mean(X,2),1,size(X,2));
        AA=alpha*eye(num_hidden_neurons)+X*L*X';
        BB=X_center*X_center'+1e-10*eye(size(X,1));
        [E,V] = eigs(AA,BB,d,'sm',opts);
        norm_term=X_center'*E;
        W=bsxfun(@times,E,sqrt(1./sum(norm_term.*norm_term)));
    else
        X_center = X-repmat(mean(X,2),1,size(X,2));
        AA=alpha*eye(num)+L*(X'*X);
        BB=pinv(X'*X)*X'*(X_center*X_center')*X+1e-10*eye(size(X,2));
        [E,V] = eigs(AA,BB,d,'sm',opts);
        norm_term=X_center'*X*E;
        W=bsxfun(@times,X*E,sqrt(1./sum(norm_term.*norm_term)));
    end
    
    
    % Update F
    [F, ~, ~]=eig1(L, c, 0);


    % Update W
    distf = L2_distance_1(F',F');
    
    distx_p = L2_distance_1(W'*X,W'*X);
    distx_p_norm_v = sqrt(sum(abs(distx_p).^2,2));
    distx_p_norm_m = sqrt(distx_p_norm_v*distx_p_norm_v');
    distx_p_norm = distx_p./distx_p_norm_m;
    distx = distx_x_norm.*distx_p_norm;
    
    A = zeros(num);
    for i=1:num
        if islocal == 1
            idxa0 = idx(i,2:k+1);
        else
            idxa0 = 1:num;
        end;
        dfi = distf(i,idxa0);
        dxi = distx(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;
    
    
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    
    %%% Check rank to decide whether to continue, repeat or stop

    [~, ~, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > 1e-11
        lambda = 2*lambda;
    elseif fn2 < 1e-11
        lambda = lambda/2; A = A_old;
    else
        break;
    end
    
end
error=0;
[clusternum, y]=graphconncomp(sparse(A)); y = y';
if clusternum ~= c
    sprintf('Can not find the correct cluster number: %d', c)
    error=1;
end;



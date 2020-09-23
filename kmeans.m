%Firstly, we open a new lovely white script in our MATLAB.
clc;
clear all;
close all;

for i=1:10

    % Set algorithm parameters
    TOL = 0.0004;
    % The max iteration
    ITER = 30;
    kappa = 4;                                                                                                     ---- cluster sayısı :))

    % The dataset that we found on UCI
    X = xlsread("perfume_data.xlsx");                                                                  ------------ Dataseti okuma

    % Called the K-means function    
    tic;
    [C, I, iter] = myKmeans(X, kappa, ITER, TOL); ------ hata toleransı
    toc
    
    % Show number of iteration taken by k-means
    disp(['k-means instance took ' int2str(iter) ' iterations to complete']);               ------------- Tamamlanan Number of iterationları gösteriyo ve int den str çeviriyo
    
    % We choose colors for the points in the resulting clustering plot
    colors = {'red', 'green', 'blue', 'black'};
    
    % Show plot of clustering
    figure;
    for i=1:kappa
       hold on, plot(X((I == i), 1), X(I == i, 2), 'p', 'color', colors{i});
    end
    title 'Perfume Data';
    xlabel 'Odor intensity';
    ylabel 'Odor quality';
    legend('Cluster 1','Cluster 2','Cluster 3','Data','Location','Best');
    % Wait key
    pause;
end
% Pause and close all windows in the end.
pause;
close all;

function [C, I, iter] = myKmeans(X, K, maxIter, TOL)

% Number of vectors in X
[vectors_num, dim] = size(X);

% Compute a random permutation of all input vectors
R = randperm(vectors_num);

% Construct indicator matrix (each entry corresponds to the cluster
% of each point in X)
I = zeros(vectors_num, 1);

% Construct centers matrix
C = zeros(K, dim);

% Take the first K points in the random permutation as the center sead
for k=1:K
    C(k,:) = X(R(k),:);
end

% iteration count
iter = 0;

% compute new clustering while the cumulative intracluster error in kept
% below the maximum allowed error, or the iterative process has not
% exceeded the maximum number of iterations permitted
while 1
    % find closest point
    for n=1:vectors_num
        % find closest center to current input point
        minIdx = 1;
        minVal = norm(X(n,:) - C(minIdx,:), 1);                                                  ----- uzunluklarını alıyo centroid seçilenle diğer noktaların
        for j=1:K
            dist = norm(C(j,:) - X(n,:), 1);
            if dist < minVal
                minIdx = j;
                minVal = dist;
            end
        end
        
        % assign point to the cluster center
        I(n) = minIdx;
    end
    
    % compute centers
    for k=1:K
        C(k, :) = sum(X(I == k, :));
        C(k, :) = C(k, :) / length(find(I == k));
    end
    
    % compute RSS error
    RSS_error = 0;
    for idx=1:vectors_num
        RSS_error = RSS_error + norm(X(idx, :) - C(I(idx),:), 2);
    end
    RSS_error = RSS_error / vectors_num;
    
    % increment iteration
    iter = iter + 1;
    
    % check stopping criteria
    if 1/RSS_error < TOL
        break;
    end
    
    if iter > maxIter
        iter = iter - 1;
        break;
    end
end
end


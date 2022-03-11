function acc = compute_accuracy(X_tr, y_tr, X_va, y_va, W, method)

total = length(y_va);
correct = zeros(5,1);   % 1NN, 3NN, 5NN, 7NN, 9NN 

for i=1: length(y_va)
    % for each validation data, compute the best label. 
    dist = zeros(length(X_tr),1);
    query = X_va{i};
   
    for j=1:length(y_tr)
        data = X_tr{j};
        if method == 1
            [dist(j),~] = mdtw_c_path(sigmoid_m(query*W), sigmoid_m(data*W),10);
        else
            [dist(j),~] = mdtw_c_path(query*W, data*W,10);
        end
    end
    
    k = length(unique(y_tr));
    [sortedValues, sortedIndexs] = sort(dist);
    labels = zeros(5,1);
    labels(1) = get_best_label(y_tr(sortedIndexs(1:1)), k);
    labels(2) = get_best_label(y_tr(sortedIndexs(1:3)), k);
    labels(3) = get_best_label(y_tr(sortedIndexs(1:5)), k);
    labels(4) = get_best_label(y_tr(sortedIndexs(1:7)), k);
    labels(5) = get_best_label(y_tr(sortedIndexs(1:9)), k);
    
    for j=1:5
        if labels(j) == y_va(i)
            correct(j) = correct(j) + 1;
        end
    end
end

acc = correct/total; 
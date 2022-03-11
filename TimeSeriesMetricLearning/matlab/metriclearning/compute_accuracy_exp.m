function acc = compute_accuracy_exp(X_tr, y_tr, X_va, y_va, W, method)

total = length(y_va);
correct = 0;

for i=1: length(y_va)
    % for each validation data, compute the best label. 
    query = X_va{i};
    bestlabel = 1;
    mindist = 999999;
    for j=1:length(y_tr)
        data = X_tr{j};
        M = create_pmatrix(length(query),length(data));
        if method == 1
            dist = expected_distance(query, data, length(query), length(data), M, W, 1);
        else
            dist = expected_distance(query, data, length(query), length(data), M, W, 0);
        end
        
        if dist < mindist
            mindist = dist;
            bestlabel = y_tr(j);
        end
    end
    
    if bestlabel == y_va(i)
        correct = correct + 1; 
    end
end

acc = correct/total;
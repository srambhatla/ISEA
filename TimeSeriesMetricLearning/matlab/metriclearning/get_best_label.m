function bestlabel = get_best_label(y, k)
%  returns mode of array. But this will return 
%  all of the tied values, not just only one. 

cc = zeros(k,1);
for i=1:length(y)
    cc(y(i)) = cc(y(i)) + 1;
end

max_val = max(cc);

vals = [];
idx = 1;
% return all the values equal to max_val
for i=1:k
    if max_val == cc(i)
        vals(idx) = i;
        idx = idx + 1;
    end
end

% get the value appears first. 
for i=1:length(y)
    if ismember(y(i),vals)
        bestlabel = y(i);
        break;
    end
end


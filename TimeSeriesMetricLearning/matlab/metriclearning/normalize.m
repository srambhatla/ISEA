function X = normalize(X)
n = length(X);
for i=1:n
    X{i}=X{i}/100;
end
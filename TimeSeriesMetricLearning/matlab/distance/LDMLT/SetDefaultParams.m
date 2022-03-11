function s = SetDefaultParams(s)
% s = SetDefaultParams(s);
% Sets default parameters
% s: user-specified parameters that are used instead of defaults


if (isfield(s, 'tripletsfactor') == 0),
    s.tripletsfactor= 10;         % using the dynamic triplets selection strategy
end

if (isfield(s, 'cycle') == 0),
    s.cycle= 10;
end

if (isfield(s, 'alphafactor') == 0),
    s.alphafactor= 2;
end
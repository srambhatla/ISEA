% analyze results. 

dataset_name = '_synthetic_nl';

% baseline
acc = zeros(5,5);
for i=1:5
    file_name = strcat('base_acc',num2str(i), dataset_name, '.mat' );
    load(file_name);
    acc(i,:) = base_acc_tst;
end
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('baseline accuracy for test data is :');
disp('mean accuracy: ');
mean(acc)
disp('std: ');
std(acc)

% linear metric
acc = zeros(5,5);
for i=1:5
    file_name = strcat('linear',num2str(i), dataset_name, '.mat' );
    load(file_name);
    acc(i,:) = linear_acc_tst;
end
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('linear metric accuracy for test data is :');
disp('mean accuracy: ');
mean(acc)
disp('std: ');
std(acc)


% non-linear metric
acc = zeros(5,5);
for i=1:5
    file_name = strcat('non_linear',num2str(i), dataset_name, '.mat' );
    load(file_name);
    acc(i,:) = non_linear_acc_tst;
end
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('non linear metric accuracy for test data is :');
disp('mean accuracy: ');
mean(acc)
disp('std: ');
std(acc)


% linear_expected metric
acc = zeros(5,5);
for i=1:5
    file_name = strcat('linear_expected',num2str(i), dataset_name, '.mat' );
    load(file_name);
    acc(i,:) = linear_expected_acc_tst;
end
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('expected linear metric accuracy for test data is :');
disp('mean accuracy: ');
mean(acc)
disp('std: ');
std(acc)

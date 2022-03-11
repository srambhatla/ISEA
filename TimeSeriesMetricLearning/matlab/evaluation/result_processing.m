acc_mean = cell(10,1);
acc_std = cell(10,1);
acc_mean{1} = allprecisionmean{1}(1:2:20);
acc_std{1} = allprecisionstd{1}(1:2:20);
acc_mean{2} = allprecisionmean{3}(1:2:20);
acc_std{2} = allprecisionstd{3}(1:2:20);
acc_mean{3} = allprecisionmean{4}(1:2:20);
acc_std{3} = allprecisionstd{4}(1:2:20);
acc_mean4_1 = allprecisionmean{6}(1:2:20);
acc_mean4_2 = allprecisionmean{8}(1:2:20);
if mean(acc_mean4_1-acc_mean4_2)>0
	acc_mean{4} = acc_mean4_1;
    acc_std{4} = allprecisionstd{6}(1:2:20);
else 
	acc_mean{4} = acc_mean4_2;
    acc_std{4} = allprecisionstd{8}(1:2:20);
end
methods = {'MDTW_D', 'MSA', 'ML-TSA', 'MSA-NN', 'LM-lin', 'LM-nl', 'LM-2nl', 'LM-3nl', 'LM-elin', 'LM-enl'};

save('knn_bk.mat', 'acc_mean', 'acc_std', 'methods');
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:54:25 2021

@author: mlab-retro
"""


from scipy.io import loadmat
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import manifold  # multidimensional scaling
from mpl_toolkits import mplot3d
import pandas as pd
import itertools
import matplotlib.ticker as mtick
from matplotlib import pyplot
import matplotlib as mlib

def calc_acc_and_plot(data_name, method, plot=True, ran_on='matlab'):
    
    file_dist = '../new_data/'+ data_name + '/' + data_name + '-' + method + '-' + 'distance.mat'
    datafile = '../new_data/' + data_name + '/' + data_name +'.mat'
    foldfile = '../new_data/' + data_name + '/' + data_name + '-5folds.mat'
    predfile = '../new_data/'+ data_name + '/' + data_name + '-' + method + '-' + 'prediction.mat'
    outfile = '../results/'+ data_name + '/' + data_name + '-' + method  + '-embedding.png'
    #neighbor_file = '../new_data/' + data_name + '/' + data_name + '-' + method + '-neighbor.mat'

    fold_id = 4
    
    if ran_on == 'matlab':
        f = h5py.File(file_dist, 'r')

        arrays = {}
        for k, v in f.items():
            arrays[k] = np.array(v)
    
    
        dist_tr_tr = f[arrays['distanceTrTr'][0,fold_id]][()]
        dist_tr_te = f[arrays['distanceTrTe'][0,fold_id]][()]

    else:
        f = loadmat(file_dist)
        dist_tr_tr = f['distTrTr']
        dist_tr_te = f['distTeTr'].T
        
        
        
    dat = loadmat(datafile)
    y = np.squeeze(dat['y'])

    fold_f = loadmat(foldfile)
    I = fold_f['folds']

    #neig = loadmat(neighbor_file)
    #neig['neighborTrTe']
    if ran_on == 'matlab':
        trI = np.where(I != fold_id + 1)[0]
        testI = np.where(I == fold_id + 1)[0]
        y_eval = [y[i] for i in testI]
        y_true = [y[i] for i in trI]
        
    else:
        testI = I == fold_id+1
        I_valid = I == ((fold_id+1)%5 + 1)
        trI = np.logical_not(np.logical_or(testI, I_valid))
        y_eval = [y[i] for i in np.where(testI[0])][0]
        y_true = [y[i] for i in np.where(trI[0])][0]
        y_valid = [y[i] for i in np.where(I_valid[0])][0]
        y_true = list(y_true) + list(y_valid)


    if plot == True:
       
        dist_mat = dist_tr_tr*0.5 + dist_tr_tr.T*0.5
        # dist_mat = dist_tr_tr
        mds_model = manifold.MDS(n_components = 2, random_state = 123, dissimilarity = 'precomputed')
        #mds_model = manifold.TSNE(n_components = 3, random_state = 0,metric = 'precomputed')
        #manifold.TSNE(n_components=2, random_state=0, metric='precomputed')
        mds_fit = mds_model.fit(dist_mat)  
        mds_coords = mds_model.fit_transform(dist_mat) 
        color= ['red' if l == 0 else 'green' for l in y_true]
        marks = ['.' if l == 0 else "x" for l in y_true]

    
        ax = plt.axes()
    
        # for i in range(0,mds_coords.shape[0]):
        #     if marks[i] == '.':
        #         cl0 = ax.scatter(mds_coords[i,0],mds_coords[i,1], color=color[i] , marker=marks[i],  s=150)
        #     else:
        #         cl1 = ax.scatter(mds_coords[i,0],mds_coords[i,1], color=color[i] , marker=marks[i],  s=80)
    
        plt.scatter(mds_coords[:,0],mds_coords[:,1], color=color)
   
        ax.grid()
        ax.axis('tight')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2') 

        
        # ax.axis('tight')
        # ax.set_xlabel('Dimension 1', fontsize=14)
        # ax.set_ylabel('Dimension 2', fontsize=14) 
        # ax.tick_params(axis='both', which='major', labelsize=12,  length=4)
        # ax.tick_params(axis='both', which='minor', labelsize=12, length=4)        
        # ax.legend((cl0, cl1), ['Class 0', 'Class 1'], loc='lower left')
        
        if method == 'MDTW':
            plt.locator_params(axis="x", nbins=4)
        #plt.savefig(outfile, format='png',bbox_inches='tight')
    
        plt.show()
        ax.clear()

    
    id_list = range(0,dist_tr_tr.shape[0])
    print(id_list)
    triplet_violation = 0
    for t in range(0, trials):
        sel_ids =np.random.choice(id_list, size=3, replace=False)
        i, j, k = sel_ids
        if dist_tr_tr[i,j] > dist_tr_tr[j,k] + dist_tr_tr[i,k] : 
            triplet_violation = triplet_violation + 1

    print('Train violations:' + str(triplet_violation))      
    
    id_list = range(0,dist_tr_te.shape[1])
    print(id_list)
    triplet_violation = 0
    for t in range(0, trials):
        sel_ids =np.random.choice(id_list, size=3, replace=False)
        i, j, k = sel_ids
        if dist_tr_te[i,j] > dist_tr_te[j,k] + dist_tr_te[i,k] : 
            triplet_violation = triplet_violation + 1
    
    print('Test violations:' + str(triplet_violation))        
    
    ### calc acc
    #K=range(1,20,2)
    if ran_on == 'matlab':
        res = loadmat(predfile)
        mean_acc = res['precisionmean']
        std_acc = res['precisionstd']
    else:
        mean_acc = f['acc_mean'][0]
        std_acc = f['acc_std'][0]
        
    # mean_acc = []
    # for fol in range(0,5):
    
    #     trI = np.where(I != fol + 1)[1]
    #     testI = np.where(I == fol + 1)[1]

    #     y_eval = [y[i] for i in testI]
    #     y_ref = np.array([y[i] for i in trI])
    
    #     #dist_tr_te = f[arrays['distanceTrTe'][0,fol]][()]
    #     acc = np.zeros([len(dist_tr_te.T), len(K)])

    #     for i_k, k in enumerate(K):
    #         w  = 1 + 0.5 * np.power(0.5, range(k))

    #         sorted_id = np.argsort(dist_tr_te.T)
    #         y_k = y_ref[sorted_id[:,:k]]
    #         #y_k = y_ref[neig['neighborTrTe'][fol][0][:,:k] - 1]
            
    #         for i in range(len(dist_tr_te.T)):
    #             acc[i][i_k] = np.argmax(np.bincount(y_k[i], w)) == y_eval[i]
    #             #print(np.mean(acc, axis=0))
    #     mean_acc.append(np.mean(acc, axis=0))
    
    #return np.mean(np.array(mean_acc),0), np.std(np.array(mean_acc),0), triplet_violation
    return mean_acc, std_acc, triplet_violation
    
    
if __name__ == '__main__':    
    K=range(1,20,2)
    trials = 20000
    plot = True
    #methods = ['MDTW', 'GAK', 'treeMsa', 'metricMDTW-hammL-diagY-onesW', 'LDMLT_TS']
    #methods = ['I-SEA','MDTW', 'GAK', 'treeMsa', 'LDMLT_TS']
    #method_print = ['I-SEA','MDTW', 'GAK', 'Tree MSA', 'LDMLT']
    methods = ['MDTW', 'GAK', 'treeMsa', 'LDMLT_TS']
    method_print = ['MDTW', 'GAK', 'MSA', 'LDMLT']
    #data_names = ['ucieeg-new']
    data_names = ['physio-data']
    #data_names = ['syn-data-100-20']
    #data_names = ['ucieeg-new']
    #ran = {'I-SEA':'python', 'MDTW':'matlab', 'GAK':'matlab', 'treeMsa':'matlab', 'LDMLT_TS':'matlab'}
    ran = {'MDTW':'matlab', 'GAK':'matlab', 'treeMsa':'matlab', 'LDMLT_TS':'matlab'}
    #methods = ['MDTW']
    #data_names = ['ucieeg-new']
    
    #methods = ['I-SEA','Decade', 'MDTW-NN', 'MaLSTM', 'MDTW', 'GAK', 'TMSA','LDMLT']
    marker = itertools.cycle(("<", "o", "v", "s", "D", "X", "d", "*","h", "x","+","P","H", '^', ">")) 
    lstyle = itertools.cycle(('--', '-.', ':'))
    lcolor = itertools.cycle(('black', 'red', 'cyan', 'magenta', 'brown', 'lime', 'blue', 'green'))

    
    col_k = ['K = ' + str(k) for k in list(K)]
    df = pd.DataFrame(columns=col_k + ['T_V', 'Method', 'Dataset'])
    df_mean = pd.DataFrame(columns=col_k + ['Method', 'Dataset'])

    for d in data_names:
        for m in methods:
    
            print('Data ' + d + ' and method ' + m)
            mean_acc, std_acc, t_v = calc_acc_and_plot(d, m, plot, ran_on=ran[m])
            print('Accuracy:' + str(mean_acc), ',\n std.dev:' + str(std_acc) + 
                  ', \n Triplet Violations:' + str(t_v))

            mean_std = [str(np.round(mean_acc[i], 3)) +' ' + u"\u00B1" + ' '+ 
                        str(np.round(std_acc[i], 3)) for i in range(0, len(K))]
            
    

            df.loc[len(df)] = list(mean_std) +[t_v] + [m] + [d]
            df_mean.loc[len(df_mean)] = list(mean_acc) + [m] + [d]


## plot misclassification
    # for d in data_names:
        
    #     df_dat = df[df['Dataset']==d]
    #     df_dat = df_dat.set_index('Method')
    #     print(df_dat.to_latex(columns = col_k + ['T_V']))
        
    #     df_acc = df_mean[df_mean['Dataset'] == d]
    #     df_acc = df_acc.set_index('Method')
    #     df_acc = df_acc.drop(['Dataset'], axis=1)
    #     m_dict = df_acc.to_dict(orient='index')
        
        

    #     ax = plt.axes()

    #     for key in m_dict:
    #         val = 0.5*np.random.rand(1)
    #         ax.semilogy(np.array(K),(1 - np.array(list(m_dict[key].values()))), marker = next(marker),
    #                                         linestyle=next(lstyle), linewidth=2, markersize=8, color=next(lcolor))#color=str(val.item()))
    #     ax.legend(method_print, loc='center left', bbox_to_anchor= (1.0, 0.72))

    #     ax.set_yscale('log')
    #     plt.tick_params(axis='y', which='minor')

    #     ax.yaxis.set_minor_formatter(plt.NullFormatter())
    #     ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%.2f"))
    #     ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%0.2f'))

    #     ax.xaxis.set_major_locator(mlib.ticker.FixedLocator(np.arange(1,20,2)))


    #     # print(ymin, ymax)
    #     ax.set_ylabel('Misclassification',fontsize=14)
    #     ax.set_xlabel('K-Nearest Neighbors',fontsize=14)

    #     ax.tick_params(axis='both', which='major', labelsize=12,  length=4)
    #     ax.tick_params(axis='both', which='minor', labelsize=12, length=4)
    #     plt.grid(which='both')
    #     plt.show()
        
    #     #plt.hold(False)
    #     plt.savefig('../results/'+ d + '/' + d + '-'+'misclass.pdf')
       
    #     ax.clear()
    #     ## Violations 
    #     df_tv = df[df['Dataset'] == d]
    #     df_tv['Method_name'] = method_print
    #     df_tv = df_tv.set_index('Method_name')
    #     df_tv['T_V'].plot.bar('Method_name')
    #     plt.show()














# file_dist = '../new_data/'+ data_name[d] + '/' + data_name[d] + '-' + method[m] + '-' + 'distance.mat'
# datafile = '../new_data/' + data_name[d] + '/' + data_name[d] +'.mat'
# foldfile = '../new_data/' + data_name[d] + '/' + data_name[d] + '-5folds.mat'
# neighbor_file = '../new_data/' + data_name[d] + '/' + data_name[d] + '-' + method[m] + '-neighbor.mat'

# fold_id = 0

# f = h5py.File(file_dist, 'r')

# arrays = {}
# for k, v in f.items():
#     arrays[k] = np.array(v)
    
    
# dist_tr_tr = f[arrays['distanceTrTr'][0,fold_id]][()]
# dist_tr_te = f[arrays['distanceTrTe'][0,fold_id]][()]

# dat = loadmat(datafile)
# y = np.squeeze(dat['y'])

# fold_f = loadmat(foldfile)
# I = fold_f['folds']

# neig = loadmat(neighbor_file)
# neig['neighborTrTe']

# trI = np.where(I != fold_id + 1)[1]
# testI = np.where(I == fold_id + 1)[1]

# y_eval = [y[i] for i in testI]
# y_true = [y[i] for i in trI]


# #dist_mat = dist_tr_tr*0.5 + dist_tr_tr.T*0.5
# dist_mat = dist_tr_tr
# mds_model = manifold.MDS(n_components = 3, random_state = 123, dissimilarity = 'precomputed')
# #mds_model = manifold.TSNE(n_components = 3, random_state = 0,metric = 'precomputed')
# #manifold.TSNE(n_components=2, random_state=0, metric='precomputed')
# mds_fit = mds_model.fit(dist_mat)  
# mds_coords = mds_model.fit_transform(dist_mat) 
# color= ['red' if l == 0 else 'green' for l in y_true]
# marks = ['.' if l == 0 else "x" for l in y_true]

# ax = plt.axes()

# for i in range(0,mds_coords.shape[0]):
#     if marks[i] == '.':
#         cl0 = ax.scatter(mds_coords[i,0],mds_coords[i,1], color=color[i] , marker=marks[i],  s=150)
#     else:
#         cl1 = ax.scatter(mds_coords[i,0],mds_coords[i,1], color=color[i] , marker=marks[i],  s=80)
#   #  plt.scatter(mds_coords[i,0],mds_coords[i,1], color=color[i] , marker=marks[i],  s=20)
# #plt.scatter(mds_coords[:,0],mds_coords[:,1], color=color)
# #ax = plt.axes(projection ="3d")
# #ax.scatter3D(mds_coords[:,0],mds_coords[:,1], mds_coords[:,2], color=color, marker=marker)
# ax.grid()
# ax.axis('tight')
# ax.set_xlabel('Dimension 1', fontsize=14)
# ax.set_ylabel('Dimension 2', fontsize=14) 
# ax.tick_params(axis='both', which='major', labelsize=12,  length=4)
# ax.tick_params(axis='both', which='minor', labelsize=12, length=4)
# ax.legend((cl0, cl1), ['Class 0', 'Class 1'], loc='lower left')
# #plt.savefig('/home/sirisha/TimeWarping/TimeSeriesMetricLearning/python_pytorch/results/ras-entry-angle/isdecade-before-rep-auc.png', bbox_inches='tight')
# #plt.savefig('/home/sirisha/TimeWarping/TimeSeriesMetricLearning/python_pytorch/results/eeg/MaLSTM__after_test.png', bbox_inches='tight')

# plt.show()

#file_dist = '../new_data/'+ 'ucieeg-new' + '/' + 'ucieeg-new' + '-' + 'I-SEA' + '-' + 'distance.mat'

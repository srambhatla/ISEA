#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:42:38 2021

@author: mlab-retro
"""




from scipy.io import loadmat
import numpy as np
import h5py
import matplotlib.pyplot as plt


m = 1
d = 0
trials = 20000

method = ['MDTW', 'GAK', 'treeMsa', 'metricMDTW-hammL-diagY-onesW', 'LDMLT_TS']
data_name = ['ucieeg-new', 'physio-data']

file_dist = '../new_data/'+ data_name[d] + '/' + data_name[d] + '-' + method[m] + '-' + 'distance.mat'
f = h5py.File(file_dist, 'r')

arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)
    
    
dist_tr_tr = f[arrays['distanceTrTr'][0,0]][()]
dist_tr_te = f[arrays['distanceTrTe'][0,0]][()]




id_list = range(0,dist_tr_tr.shape[0])
triplet_violation = 0
for t in range(0, trials):
    sel_ids =np.random.choice(id_list, size=3, replace=False)
    i, j, k = sel_ids
    if dist_tr_tr[i,j] > dist_tr_tr[j,k] + dist_tr_tr[i,k] : 
        triplet_violation = triplet_violation + 1

print(data_name[d], method[m])
print(triplet_violation, (triplet_violation/trials)*100)


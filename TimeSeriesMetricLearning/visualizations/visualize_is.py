#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:52:05 2021

@author: mlab-retro
"""



import matplotlib.pyplot as plt
import scipy.stats as st
import random
import numpy as np
import pandas as pd

np.random.seed(0)
A = pd.DataFrame({'x':[random.uniform(0, 1) for i in range(0,100)], 'y':[random.uniform(0, 1) for i in range(0,100)], 
                  'z':[random.uniform(0, 1) for i in range(0,100)], 'w':[random.uniform(0, 1) for i in range(0,100)]})
B = pd.DataFrame({'x':[random.uniform(0, 1) for i in range(0,100)], 'y':[random.uniform(0, 1) for i in range(0,100)], 
                  'z':[random.uniform(0, 1) for i in range(0,100)], 'w':[random.uniform(0, 1) for i in range(0,100)]})



def plot_2d_kde(df, flag):
    # Extract x and y
    x = df['x']
    y = df['y']
    # Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # We will fit a gaussian kernel using the scipy’s gaussian_kde method
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
  
    if flag == 'IS':
        surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
    else:
        surf = ax.plot_surface(xx, yy, 2*np.ones(xx.shape), rstride=1, cstride=1, cmap='plasma', edgecolor='none')
    
    #ax.set_xlabel('Alignment Paths for time-series $\mathbf{X}$',  fontsize = 14)
    #ax.set_ylabel('Alignment Paths for time-series $\mathbf{Y}$',  fontsize = 14)
    #ax.zaxis.set_rotate_label(False)
    #ax.zaxis.set_rotate_label(False)
    #ax.set_zlabel('Path Similarity',  fontsize = 14, rotation=102)
    #ax.zaxis.labelpad = -10
    
    

    
    #ax.set_title('Surface plot of Gaussian 2D KDE')
 
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    

    cbar = fig.colorbar(surf, shrink=0.5, aspect=5, ticks=[]) # add color bar indicating the PDF

    ax.view_init(60, 35)
    
    sam_x = [random.uniform(0, 1) for i in range(0,100)]
    sam_y = [random.uniform(0, 1) for i in range(0,100)]
    
    func_loc = list(zip([find_func_ids(ptx, x) for ptx in sam_x ], 
                        [find_func_ids(ptx, y) for ptx in sam_y ]))
    
    print(len(func_loc))
    val = [f[l] for l in func_loc]
    #ax.scatter3D(sam_x,sam_y, val, 'k', marker='o')
    
    return sam_x,sam_y,f, x, y


def find_func_ids(ptx, x):
    idx = np.argmin(np.abs(x - ptx))
    return idx
    
if __name__ == '__main__': 
    plt.clf()
    flag = 'unif'
    x, y, f, xx, yy = plot_2d_kde(B, flag)
    outfile = '../visualizations/' + flag +'.png'
    plt.savefig(outfile, format='png',bbox_inches='tight',transparent=True)

    
    
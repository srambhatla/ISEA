from __future__ import print_function

import ctypes
import os
import sys
import time
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean

import numpy as np
from numpy.ctypeslib import ndpointer

from sdtw.path import gen_all_paths
from random import randint



def get_distance_handler(distance_name):
    '''
    Get distance handler by name.
    '''
    print('Looking for ', distance_name)
    if distance_name.startswith("Manhattan"):
        return Manhattan_LSTM()
    if distance_name.startswith("DTW_C"):
        return DTW_C()
    elif distance_name.startswith("DTW_Python"):
        return DTW_Python()
    elif distance_name.startswith("soft-dtw"):
        return Soft_DTW()
    elif distance_name.startswith('EDISTNew'):
        # Get parameters, e.g., EDIST_New_low_high_num
        fields = distance_name.split('_')
        T_low = int(fields[2])
        T_high= int(fields[3])
        sample_num = int(fields[4])
        if fields[1] == 'Python':
            return EDISTNew_Python(T_low=T_low, T_high=T_high, 
                                   sample_num=sample_num)
        elif fields[1] == 'CPython':
            return EDISTNew_CPython(T_low=T_low, T_high=T_high, 
                                    sample_num=sample_num)
        else:
            return EDISTNew_C(T_low=T_low, T_high=T_high, 
                              sample_num=sample_num)
    elif distance_name.startswith('EDISTisNew'):
        # Get parameters, e.g., EDIST_New_low_high_num
        
        fields = distance_name.split('_')
        T_low = int(fields[2])
        T_high= int(fields[3])
        sample_num = int(fields[4])
       
        return EDISTNewIS_C(T_low=T_low, T_high=T_high, 
                              sample_num=sample_num)
    
    elif distance_name.startswith('EDISTkdeIS'):
        # Get parameters, e.g., EDIST_New_low_high_num
        
        fields = distance_name.split('_')
        T_low = int(fields[2])
        T_high= int(fields[3])
        sample_num = int(fields[4])
       
        return EDISTNewISkde_C(T_low=T_low, T_high=T_high, 
                              sample_num=sample_num)
    


    else:
        print('Distance {} not found'.format(distance_name))
        raise NotImplementedError



class Distance(object):
    '''
    Class for time series distanace computation.
    Methods:
         dist_and_path(X1, X2), dist(X1, X2), path(X1, X2)
    '''
    def __init__(self, name):
        self.name = name
    
    def dist(self, X1, X2):
        '''
        Take two Time series X1, X2 with size of [T1, P], [T2, P]
        Returns:
            Value: A float number
        '''
        raise NotImplementedError
    
    def dist_and_path(self, X1, X2):
        '''
        Take two Time series X1, X2 with size of [T1, P], [T2, P]
        Returns:
            Value: A float number
            Path1: Int vector of [1, L]
            Path2: Int vector of [1, L]
        '''
        raise NotImplementedError
    
    def path(self, X1, X2):
        '''
        Take two Time series X1, X2 with size of [T1, P], [T2, P]
        Returns:
            Path1: Int vector of [1, L]
            Path2: Int vector of [1, L]
        '''
        raise NotImplementedError       
    
    def compute_dist_mtx(self, timeseries, verbose=1):
        '''
        Compute distance matrix for all given time series.
        timeseries: A list of size N, each is an np.array of [T by D]
        '''
        n_samples = len(timeseries)
        print('... compute pairwise {} distance for {} time series'.format(
                self.name, n_samples))
        timer_1 = time.time()
        dist_tr_tr = np.empty([n_samples, n_samples], dtype='float');
        for i in range(n_samples):
            if i % 10 == 0 and verbose:
                print(i if i % 100 == 0 else '.', end=''),
                sys.stdout.flush()
            for j in range(i+1):
                dist_ret = self.dist(timeseries[i], timeseries[j])
                dist_tr_tr[i][j] = dist_ret
                dist_tr_tr[j][i] = dist_tr_tr[i][j]
        sys.stdout.flush()
        print('{} ran for {}m'.format(self.name, (time.time() - timer_1) / 60.))
        return dist_tr_tr        


class DTW(Distance):
    def __init__(self, **kwargs):
        super(DTW, self).__init__(**kwargs)
        
    def dist(self, X1, X2):
        return self.dist_and_path(X1, X2)[0]
    
    def path(self, X1, X2):
        return self.dist_and_path(X1, X2)[1:3]
    

class DTW_C(DTW):
    def __init__(self, libpath=None):
        super(DTW_C, self).__init__(name='DTW_C')
        if libpath is None:
            libpath = './mdtw_c_path.so' if os.name == 'posix' \
                    else './mdtw_c_path.dll'
        self.lib = ctypes.cdll.LoadLibrary(libpath)
        self.fun = self.lib.mdtw_c
        self.fun.restype = None
        self.fun.argtypes = [
            ndpointer(ctypes.c_float), ctypes.c_int, # float *X1, int t1
            ndpointer(ctypes.c_float), ctypes.c_int, # float *X2, int t2
            ctypes.c_int, ctypes.c_int,     # int p, int w
            ctypes.POINTER(ctypes.c_float), # float *d,
            ctypes.POINTER(ctypes.c_int),   # int *cntR,
            ndpointer(ctypes.c_int)        # int *dpath
        ]

    def dist_and_path(self, X1, X2):
        '''
        Args:
            X1, X2: Two time series with same dimension P: [M,P] and [N,P].
        Returns:
            dist.value: A float number, the distance between X1 and X2.
            path(s): Two lists of length [L]. L is the length of the DTW path.
        '''
        M, N=len(X1), len(X2)
        P = len(X1[0])
        path = np.empty([2, M+N], dtype=np.int32)
        dist = ctypes.c_float()    
        p_length = ctypes.c_int()
        X1 = X1.astype(dtype=np.float32)
        X2 = X2.astype(dtype=np.float32)
        self.fun(X1, M, X2, N, P, 10, ctypes.byref(dist), 
                 ctypes.byref(p_length), path)
        return dist.value, path[0, :p_length.value], path[1, :p_length.value]


class DTW_Python(DTW):
    def __init__(self):
        super(DTW_Python, self).__init__(name='DTW_Python')
        
    def dist_and_path(self, X1, X2):
        vector_dist = lambda v1, v2: np.sum(np.square(v1-v2))
        M, N = len(X1), len(X2)
        cost = np.zeros((M, N))
        prev = np.zeros((M, N))
        cost[0][0]= vector_dist(X1[0], X2[0])
        for i in range(1, M):
            cost[i][0] = cost[i-1][0] + vector_dist(X1[i], X2[0])
        for j in range(1, N):
            cost[0][j] = cost[0][j-1] + vector_dist(X1[0], X2[j])
        for i in range(1,M):
            for j in range(1,N):
                choices =[cost[i-1, j-1], cost[i, j-1], cost[i-1, j]]
                prev[i, j] = np.argmin(choices) + 1
                cost[i, j] = np.amin(choices) + vector_dist(X1[i], X2[j])
        path = []
        i, j = M-1, N-1
        while i > 0 and j > 0:
            path.append([i, j])
            if prev[i,j] == 1:
                i, j = i-1, j-1 
            elif prev[i,j] == 2:
                j=j-1
            else:  #prev[i,j] == 3
                i=i-1
        while i > 0:
            path.append([i, j])
            i = i-1
        while j > 0:
            path.append([i, j])
            j = j-1
        path.append([0,0])
        path.reverse()
        path = np.array(path)
        return cost[-1, -1], path[0, :], path[1, :]


class ExpDis(Distance):
    def __init__(self, T_low, T_high, sample_num=1, **kwargs):
        '''
        If sample num > 1, we will use $sample_num$ samples to approximate the 
        expected distance.
        '''
        super(ExpDis, self).__init__(**kwargs)
        self.T_low = T_low
        self.T_high = T_high
        self.sample_num = max(sample_num, 1)


class EDISTNew_C(ExpDis):
    def __init__(self, libpath=None, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'EDISTNew_C_{}_{}_{}'.format(
                    kwargs['T_low'], kwargs['T_high'], kwargs['sample_num'])
        super(EDISTNew_C, self).__init__(**kwargs)
        if libpath is None:
            libpath = './edist_c_path.so' if os.name == 'posix' \
                else './edist_c_path.dll'
        self.lib = ctypes.cdll.LoadLibrary(libpath)
        self.fun = self.lib.edist_c
        self.fun.restype = None
        self.fun.argtypes = [
            ndpointer(ctypes.c_float), ctypes.c_int,    # float *X1, int t1
            ndpointer(ctypes.c_float), ctypes.c_int,    # float *X2, int t2
            ctypes.c_int,                   # int p, 
            ctypes.c_int, ctypes.c_int,     # int T_low, int T_high
            ctypes.c_int,                   # int n_sample
            ctypes.POINTER(ctypes.c_float), # float *d
            ctypes.POINTER(ctypes.c_int),   # int *cntR
            ndpointer(ctypes.c_int)         # int *dpath
        ]
        self.fun_path = self.lib.path_c
        self.fun_path.restype = None
        self.fun_path.argtypes = [
            ctypes.c_int, ctypes.c_int,     # int t1, int t2
            ctypes.c_int, ctypes.c_int,     # int T_low, int T_high
            ctypes.c_int,                   # int n_sample
            ctypes.POINTER(ctypes.c_int),   # int *cntR
            ndpointer(ctypes.c_int)         # int *dpath
        ]
        
    def dist_and_path(self, X1, X2):
        '''
        Args:
            X1, X2: Two time series with same dimension P: [M,P] and [N,P].
        Returns:
            dist.value: A float number, the distance between X1 and X2.
            path(s): Two lists of length [L]. L is the sum of length of paths.
        '''
        M, N = len(X1), len(X2) #sirisha
        #M, N = X1.size, X2.size #sirisha
        
        P = len(X1[0]) #sirisha
        #P = X1[0].size #sirisha
        
        path = np.empty([2, self.sample_num*self.T_high], dtype=np.int32)
        dist = ctypes.c_float()    
        p_length = ctypes.c_int()
        #p_length = ctypes.c_float()
        self.fun(X1, M, X2, N, P, self.T_low, self.T_high, self.sample_num, 
                 ctypes.byref(dist), ctypes.byref(p_length), path)
        return dist.value, path[0, :p_length.value], path[1, :p_length.value]

    def dist(self, X1, X2):
        return self.dist_and_path(X1, X2)[0]
    
    def path(self, X1, X2):
        T1, T2 = len(X1), len(X2)
        path = np.empty([2, self.sample_num*self.T_high], dtype=np.int32)
        p_length = ctypes.c_int()
        self.fun_path(T1, T2, self.T_low, self.T_high, self.sample_num, 
                 ctypes.byref(p_length), path)
        return path[0, :p_length.value], path[1, :p_length.value]


class EDISTNew_Python(ExpDis):
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'EDISTNew_Python_{}_{}_{}'.format(
                    kwargs['T_low'], kwargs['T_high'], kwargs['sample_num'])
        super(EDISTNew_Python, self).__init__(**kwargs)

    def sample_single_path(self, T, L):
        '''
        Select L numbers from T different candidates, each candidate can be
        selected for 0 or more times.
        It is the same as (T+L-1, T-1). That is, to find T-1 separators.
        TODO: Any better implementation which is more effieicient?
        Args:
            T: Length of the original time series.
            L: Length of the sampled path.
        Return:
            A list of length L.
        '''
        pos = sorted(np.random.choice(T+L-1, size=T-1, replace=False))
        temp = -1
        solution = [-1] * T
        for i, p in enumerate(pos):
            solution[i] = pos[i] - temp - 1
            temp = pos[i]
        solution[T-1] = L + T - 2 - pos[T-2]
        result = []
        for i in range(T):
            result = result + [i] * solution[i]
        return result
        
    def sample_path(self, T1, T2):
        '''
        Returns:
            Two lists of length [L]. L ~ Unif(self.T_low, self.T_high)
        '''
        L = np.random.randint(self.T_low, self.T_high)
        path1 = self.sample_single_path(T1, L)
        path2 = self.sample_single_path(T2, L)
        return path1, path2, L
        
    def compute_path_distance(self, X1, X2, path1, path2, L):
        X1_selected = X1[path1, :]
        X2_selected = X2[path2, :]
        return np.sum((X1_selected - X2_selected)**2) / L
        
    def dist_and_path(self, X1, X2):
        T1, T2 = len(X1), len(X2)
        path1, path2 = [], []
        distance = 0.
        for i in range(self.sample_num):
            path1_i, path2_i, L = self.sample_path(T1, T2)
            distance += self.compute_path_distance(X1, X2, path1, path2, L)
            path1.extend(path1_i)
            path2.extend(path2_i)
        distance /= self.sample_num
        return distance, path1, path2
    
    def dist(self, X1, X2):
        return self.dist_and_path(X1, X2)[0]
    
    def path(self, X1, X2):
        T1, T2 = len(X1), len(X2)
        path1, path2 = [], []
        for i in range(self.sample_num):
            path1_i, path2_i, _ = self.sample_path(T1, T2)
            path1.extend(path1_i)
            path2.extend(path2_i)
        return path1, path2



class EDISTNew_CPython(EDISTNew_Python):
    def __init__(self, libpath=None, **kwargs):
        kwargs['name'] = 'EDISTNew_CPython_{}_{}_{}'.format(
                    kwargs['T_low'], kwargs['T_high'], kwargs['sample_num'])        
        super(EDISTNew_CPython, self).__init__(**kwargs)
        if libpath is None:
            libpath = './edist_c_path.so' if os.name == 'posix' \
                else './edist_c_path.dll'
        self.lib = ctypes.cdll.LoadLibrary(libpath)
        self.fun = self.lib.edist_cp
        self.fun.restype = None
        # X1, t1, X2, t2, p, n_sample, t_list, **paths1, **paths2, *dist)
        self.fun.argtypes = [
            ndpointer(ctypes.c_float), ctypes.c_int,   # float *X1, int t1 
            ndpointer(ctypes.c_float), ctypes.c_int,   # float *X2, int t2
            ctypes.c_int, ctypes.c_int,   # int p, int n_sample
            ndpointer(ctypes.c_int),   # int *t_list
            ndpointer(ctypes.c_int),   # int *dpath1
            ndpointer(ctypes.c_int),   # int* dpath2
            ctypes.POINTER(ctypes.c_float)   # float *d
        ]

    def dist_and_path(self, X1, X2):
        T1, T2 = len(X1), len(X2)
        P = len(X1[0])
        path1, path2 = [], []
        Ls = np.zeros([self.sample_num], dtype='int32')
        for i in range(self.sample_num):
            path1_i, path2_i, Ls[i] = self.sample_path(T1, T2)
            path1.extend(path1_i)
            path2.extend(path2_i)        
        dist = ctypes.c_float()
        # X1, t1, X2, t2, p, n_sample, t_list, **paths1, **paths2, *dist)
        self.fun(X1, T1, X2, T2, P, self.sample_num, Ls,
                 np.asarray(path1, dtype='int32'), 
                 np.asarray(path2, dtype='int32'),
                 ctypes.byref(dist))
        return dist.value, path1, path2


class Manhattan_LSTM(Distance):
    def __init__(self):
        super(Manhattan_LSTM, self).__init__('Manhattan')

    def dist(self, X1, X2):
        path1, path2 = self.path(X1, X2)
        pX1 = [X1[i] for i in path1]
        pX2 = [X2[i] for i in path2]
        assert(len(pX1)==len(pX2))
        cost = 0.0
        for i in range(len(pX1)):
            cost+=abs(pX1[i]-pX2[i])
        return np.sum(cost) # sirisha
        #return cost
    
    def path(self, X1, X2):
        
        mat = gen_all_paths(len(X1),len(X2))
        #for i in range(0, randint(0, max(len(X1), len(X2)))):
        next(mat)
        a, b = np.nonzero(next(mat))
        return a, b
        
        #return np.arange(len(X1)), np.arange(len(X2))
    
    def dist_and_path(X1, X2):
        
        path1, path2 = self.path(X1, X2)
        return self.dist(X1, X2), path1, path2

    
class EDISTNewIS_C(ExpDis):
    def __init__(self, libpath=None, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'EDISTNewIS_C_{}_{}_{}'.format(
                    kwargs['T_low'], kwargs['T_high'], kwargs['sample_num'])
        super(EDISTNewIS_C, self).__init__(**kwargs)
        if libpath is None:
            # Without KDE
            libpath = './edist_is_c_path.so' if os.name == 'posix' \
                else './edist_is_c_path.dll'
            #With KDE
#             libpath = './kde-edist_is_c_path_py.so' if os.name == 'posix' \
#                 else './kde-edist_is_c_path_py.dll'
           
    
        self.lib = ctypes.cdll.LoadLibrary(libpath)
        self.fun = self.lib.edist_IS_c
        self.fun.restype = None
        self.fun.argtypes = [
            ndpointer(ctypes.c_float), ctypes.c_int,    # float *X1, int t1
            ndpointer(ctypes.c_float), ctypes.c_int,    # float *X2, int t2
            ctypes.c_int,                   # int p, 
            ctypes.c_int, ctypes.c_int,     # int T_low, int T_high
            ctypes.c_int,                   # int n_sample
            ctypes.POINTER(ctypes.c_float), # float *d
            ctypes.POINTER(ctypes.c_int),   # int *cntR
            ndpointer(ctypes.c_int)         # int *dpath
        ]
        self.fun_path = self.lib.path_c
        self.fun_path.restype = None
        self.fun_path.argtypes = [
            ctypes.c_int, ctypes.c_int,     # int t1, int t2
            ctypes.c_int, ctypes.c_int,     # int T_low, int T_high
            ctypes.c_int,                   # int n_sample
            ctypes.POINTER(ctypes.c_int),   # int *cntR
            ndpointer(ctypes.c_int)         # int *dpath
        ]
        
    def dist_and_path(self, X1, X2):
        '''
        Args:
            X1, X2: Two time series with same dimension P: [M,P] and [N,P].
        Returns:
            dist.value: A float number, the distance between X1 and X2.
            path(s): Two lists of length [L]. L is the sum of length of paths.
        '''
        M, N = len(X1), len(X2) #sirisha
        #M, N = X1.size, X2.size #sirisha
        #print(X1.shape)
        P = len(X1[0]) #sirisha
        #P = X1[0].shape[] #sirisha
        path = np.empty([2, self.sample_num*self.T_high], dtype=np.int32)
        dist = ctypes.c_float()    
        p_length = ctypes.c_int()
        self.fun(X1, M, X2, N, P, self.T_low, self.T_high, self.sample_num, 
                 ctypes.byref(dist), ctypes.byref(p_length), path)
        return dist.value, path[0, :p_length.value], path[1, :p_length.value]

    def dist(self, X1, X2):
        return self.dist_and_path(X1, X2)[0]
    
    def path(self, X1, X2):
        T1, T2 = len(X1), len(X2)
        path = np.empty([2, self.sample_num*self.T_high], dtype=np.int32)
        p_length = ctypes.c_int()
        self.fun_path(T1, T2, self.T_low, self.T_high, self.sample_num, 
                 ctypes.byref(p_length), path)
        return path[0, :p_length.value], path[1, :p_length.value]
    
    
    

class EDISTNewISkde_C(ExpDis):
    def __init__(self, libpath=None, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'EDISTNewISkde_C_{}_{}_{}'.format(
                    kwargs['T_low'], kwargs['T_high'], kwargs['sample_num'])
        super(EDISTNewISkde_C, self).__init__(**kwargs)
        if libpath is None:
            # Without KDE
#             libpath = './edist_is_c_path.so' if os.name == 'posix' \
#                 else './edist_is_c_path.dll'
            #With KDE
            libpath = './kde-edist_is_c_path_py.so' if os.name == 'posix' \
                else './kde-edist_is_c_path_py.dll'
           
    
        self.lib = ctypes.cdll.LoadLibrary(libpath)
        self.fun = self.lib.edist_IS_c
        self.fun.restype = None
        self.fun.argtypes = [
            ndpointer(ctypes.c_float), ctypes.c_int,    # float *X1, int t1
            ndpointer(ctypes.c_float), ctypes.c_int,    # float *X2, int t2
            ctypes.c_int,                   # int p, 
            ctypes.c_int, ctypes.c_int,     # int T_low, int T_high
            ctypes.c_int,                   # int n_sample
            ctypes.POINTER(ctypes.c_float), # float *d
            ctypes.POINTER(ctypes.c_int),   # int *cntR
            ndpointer(ctypes.c_int)         # int *dpath
        ]
        self.fun_path = self.lib.path_c
        self.fun_path.restype = None
        self.fun_path.argtypes = [
            ctypes.c_int, ctypes.c_int,     # int t1, int t2
            ctypes.c_int, ctypes.c_int,     # int T_low, int T_high
            ctypes.c_int,                   # int n_sample
            ctypes.POINTER(ctypes.c_int),   # int *cntR
            ndpointer(ctypes.c_int)         # int *dpath
        ]
        
    def dist_and_path(self, X1, X2):
        '''
        Args:
            X1, X2: Two time series with same dimension P: [M,P] and [N,P].
        Returns:
            dist.value: A float number, the distance between X1 and X2.
            path(s): Two lists of length [L]. L is the sum of length of paths.
        '''
        M, N = len(X1), len(X2) #sirisha
        #M, N = X1.size, X2.size #sirisha
        #print(X1.shape)
        P = len(X1[0]) #sirisha
        #P = X1[0].shape[] #sirisha
        path = np.empty([2, self.sample_num*self.T_high], dtype=np.int32)
        dist = ctypes.c_float()    
        p_length = ctypes.c_int()
        self.fun(X1, M, X2, N, P, self.T_low, self.T_high, self.sample_num, 
                 ctypes.byref(dist), ctypes.byref(p_length), path)
        return dist.value, path[0, :p_length.value], path[1, :p_length.value]

    def dist(self, X1, X2):
        return self.dist_and_path(X1, X2)[0]
    
    def path(self, X1, X2):
        T1, T2 = len(X1), len(X2)
        path = np.empty([2, self.sample_num*self.T_high], dtype=np.int32)
        p_length = ctypes.c_int()
        self.fun_path(T1, T2, self.T_low, self.T_high, self.sample_num, 
                 ctypes.byref(p_length), path)
        return path[0, :p_length.value], path[1, :p_length.value]

    
    
class Soft_DTW(Distance):
    def __init__(self):
        super(Soft_DTW, self).__init__('soft-dtw')

    def dist(self, X1, X2):
        D = SquaredEuclidean(X1, X2)
        sdtw = SoftDTW(D, gamma=1.0)
        
        value = sdtw.compute()
        return value # sirisha
        #return cost
    
    def path(self, X1, X2):
        mat = gen_all_paths(len(X1),len(X2))
        for i in range(0, randint(0, max(len(X1), len(X2)))):
            next(mat)

        a, b = np.nonzero(next(mat))
        return a, b
    
    def dist_and_path(X1, X2):
        path1, path2 = self.path(X1, X2)
        return self.dist(X1, X2), path1, path2
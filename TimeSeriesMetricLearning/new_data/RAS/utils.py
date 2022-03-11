import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ==================== Preprocessing specifics ====================#
def euler_angle_to_quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sy = np.sin(yaw * 0.5)
    sp = np.sin(pitch * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.vstack((w,x,y,z)).T

def conjugate_quaternion(quaternion):
    q = np.copy(quaternion)
    q[:,1:] = -1 * q[:,1:]
    return q

def inverse_quarternion(quaternion):
    q = np.copy(quaternion)
    q_conj = conjugate_quaternion(q)
    return q_conj / np.array([(q**2).sum(axis=1).T] * 4).T

def quaternion_multiply_singular(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def quaternion_multiply(quaternion1, quaternion0):
    q = []
    for idx in range(quaternion1.shape[0]):
        q.append(quaternion_multiply_singular(quaternion1[idx], quaternion0[idx]))
    return np.array(q)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def process_entry(X_i):
    processed_entry = np.zeros((X_i.shape[0],16))
    
    camera_xyz = X_i[:,0:3]
    camera_quad = euler_angle_to_quaternion(X_i[:,3], X_i[:,4], X_i[:,5])
    camera_quad_inv = inverse_quarternion(camera_quad)
    
    processed_entry[:,0:3] = X_i[:,6:9]-camera_xyz
    processed_entry[:,3:7] = quaternion_multiply(camera_quad_inv, euler_angle_to_quaternion(X_i[:,9], X_i[:,10], X_i[:,11]))
    processed_entry[:,7] = X_i[:,12]
    
    processed_entry[:,8:11] = X_i[:,13:16]-camera_xyz
    processed_entry[:,11:15] = quaternion_multiply(camera_quad_inv, euler_angle_to_quaternion(X_i[:,16], X_i[:,17], X_i[:,18]))
    processed_entry[:,15] = X_i[:,19]
    
    average_window = 20
    if X_i.shape[0]-average_window+1 <= 0:
        return processed_entry
    
    average_entry = np.zeros((X_i.shape[0]-average_window+1,16))
    for i in range(16):
        average_entry[:,i] = moving_average(processed_entry[:,i], average_window)
    
    return average_entry

def log2timedict(logfile):
    with open(logfile, 'r') as f:
        data = f.readlines()
        
    start_time = None
    timedict = {}
    cur_time = 0.0
    for line in data:
        line = line.strip()
        if "Time Stamp:" in line:
            if start_time is None:
                start_time = float(line.strip().split()[-1])
            else:
                cur_time = float(line.strip().split()[-1]) - start_time
            timedict[cur_time] = []
        else:
            timedict[cur_time].append(float(line.strip().split()[-1]))
    return timedict

def record_info(info, filename, mode):
    if mode =='train':
        result = (
              'Time: {time} '
              'Loss: {loss} '
              'Prec@1: {top1} '
              'AUC: {auc} '
              'LR {lr}\n'.format(time=info['Time'], loss=info['Loss'], top1=info['Prec@1'], auc=info['AUC'], lr=info['lr']))      
        print(result)

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Time','Loss','Prec@1','AUC', 'lr']
        
    if mode =='test':
        result = (
              'Time: {time} '
              'Loss: {loss} '
              'Prec@1: {top1} '
              'AUC: {auc}\n'.format(time=info['Time'],loss=info['Loss'], top1=info['Prec@1'], auc=info['AUC']))      
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Time','Loss','Prec@1', 'AUC']
    
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names)   

# ==================== Image utils ====================#
def tensor2img(tensor):
    return tensor.permute(1,2,0).cpu().detach().numpy()

def img2uint8(array):
    return (array * 255).astype(np.uint8)

def tensor2uint8(tensor):
    return img2uint8(tensor2img(tensor))


# ==================== General utils ====================#
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
def makepath(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)  
        
def save_checkpoint(state, checkpoint_path):
    makepath(checkpoint_path)
    torch.save(state, checkpoint_path)
    
def save_prediction(preds, pred_path):
    makepath(pred_path)
    with open(pred_path,'wb') as f: pickle.dump(preds, f)
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

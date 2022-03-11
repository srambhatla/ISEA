import os
import pickle
import random

from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpatialDataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_rgb_image(self, video_name, frame_index):
        path = os.path.join(self.root_dir, video_name, 'frame' + str(frame_index).zfill(6) + '.jpg')
        img = Image.open(path)
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):
        label = self.values[idx]
    
        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            frame_indicies = []
            frame_indicies.append(random.randint(1, int(nb_clips/3)))
            frame_indicies.append(random.randint(int(nb_clips/3), int(nb_clips*2/3)))
            frame_indicies.append(random.randint(int(nb_clips*2/3), int(nb_clips)))
            
            data = {}
            for i, frame_index in enumerate(frame_indicies):
                data['img'+str(i)] = self.load_rgb_image(video_name, frame_index)
            sample = (video_name, data, label)
            
        elif self.mode == 'val':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.load_rgb_image(video_name, int(frame_index))
            sample = (video_name, data, label)
            
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class SpatialDataloader():
    def __init__(self, batch_size, num_workers, data_path, seed):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.task_name = self.data_path.split('/')[-1]
        self.frame_count = pickle.load(open('../multi_modal/processed_data/videos/frame_counts/{}_frame_count.p'.format(self.task_name),'rb'))  ## <path_to_data>
        self.label = pickle.load(open('../multi_modal/processed_data/videos/labels/{}_label.p'.format(self.task_name),'rb')) ## <path_to_data>
                                 
        videos_to_ignore = [video for video in self.frame_count.keys() if self.frame_count[video] < 12]
        for video in videos_to_ignore:
            if video in self.frame_count: del self.frame_count[video]
            if video in self.label: del self.label[video]
        
        # split the training and testing videos
        self.train_video, self.test_video, _, _ = train_test_split(list(self.label.keys()), list(self.label.values()), test_size=0.2, random_state=self.seed)
        self.test_video = {video:self.label[video] for video in self.test_video}
            
    def get_training_dic(self):
        ''' 
        Training dict = {'video_name num_max_frame': label}
        '''
        self.dic_training={}
        for video in self.train_video:
            nb_frame = self.frame_count[video] - 10 + 1
            key = '{} {}'.format(video, str(nb_frame))
            label = self.label[video]
            self.dic_training[key] = int(label)
            
    def val_sample30(self):
        ''' 
        Testing dict = {'video_name frame_number': label}
        Evenly spaced every 30 frames
        '''
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video] - 10 + 1
            
            if nb_frame < 30:
                num_sample = nb_frame
            else:
                num_sample = 30
            interval = int(nb_frame/num_sample)
            for i in range(num_sample):
                frame = i*interval
                key = '{} {}'.format(video, str(frame+1))
                self.dic_testing[key] = int(self.label[video])    

    def get_train_loader(self):
        training_set = SpatialDataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        print('==> Training data:', len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def get_val_loader(self):
        validation_set = SpatialDataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        print('==> Validation data:', len(validation_set), 'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
    
    def run(self):
        self.get_training_dic()
        self.val_sample30()
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        return train_loader, val_loader, self.test_video
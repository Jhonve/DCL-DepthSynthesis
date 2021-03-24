import os
import random

import numpy as np
import h5py
import cv2

import torch
import torch.utils.data as data

class ImageTaskDataset(data.Dataset):
    def __init__(self, opt, data_paths_A, data_paths_B=None):
        super(ImageTaskDataset).__init__()
        self.opt = opt
        self.is_train = self.opt.isTrain

        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_threads
        self.data_paths_A = data_paths_A

        if self.is_train:
            self.data_paths_B = data_paths_B

        if self.is_train:
            if len(self.data_paths_A) < len(self.data_paths_B):
                self.SIZE = len(self.data_paths_A)
            else:
                self.SIZE = len(self.data_paths_B)
        else:
            self.SIZE = len(self.data_paths_A)
        
    def __len__(self):
        return self.SIZE

    def loadColorImage(self, data_path):
        if self.is_train:
            data_path = data_path.decode('UTF-8')
        rgb_img = cv2.imread(data_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        if self.opt.zoom_out_scale != 1:
            if self.opt.zoom_out_scale == 0:
                rgb_img = cv2.resize(rgb_img, (self.opt.fixed_size, self.opt.fixed_size), interpolation=cv2.INTER_AREA)
            else:
                rgb_img = cv2.resize(rgb_img, (int(rgb_img.shape[1] / self.opt.zoom_out_scale), int(rgb_img.shape[0] / self.opt.zoom_out_scale)), interpolation=cv2.INTER_AREA)

        rgb_img = np.array(rgb_img).astype(np.float32)

        rgb_img = rgb_img / 255.
        rgb_img = rgb_img.transpose(2, 0, 1)
        return rgb_img

    def __getitem__(self, index):
        data_item = {}
        
        data_item['A'] = self.loadColorImage(self.data_paths_A[index])
        if self.is_train:
            data_item['B'] = self.loadColorImage(self.data_paths_B[index])

        data_item['A_paths'] = self.data_paths_A[index]
        if self.is_train:
            data_item['B_paths'] = self.data_paths_B[index]
        return data_item

    def getDataloader(self):
        if self.is_train:
            return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        else:
            return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)
    
class DepthDataset(data.Dataset):
    def __init__(self, opt, data_path_clean, data_path_noise=None):
        super(DepthDataset).__init__()
        self.opt = opt
        self.is_train = self.opt.isTrain
        
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_threads
        self.data_path_clean = data_path_clean

        if self.is_train:
            self.data_path_noise = data_path_noise

        if self.is_train:
            if len(self.data_path_clean) < len(self.data_path_noise):
                self.SIZE = len(self.data_path_clean)
            else:
                self.SIZE = len(self.data_path_noise)
        else:
            self.SIZE = len(self.data_path_clean)

    def __len__(self):
        return self.SIZE

    def loadRGBDImg(self, data_path):
        if self.is_train:
            data_path = data_path.decode('UTF-8')
        data_path_depth = data_path.split('|')[0]
        data_path_rgb = data_path.split('|')[1]

        depth_img = cv2.imread(data_path_depth, 2)
        rgb_img = cv2.imread(data_path_rgb)

        if self.opt.zoom_out_scale != 1:
            depth_img = cv2.resize(depth_img, (int(depth_img.shape[1] / self.opt.zoom_out_scale), int(depth_img.shape[0] / self.opt.zoom_out_scale)), interpolation=cv2.INTER_NEAREST)
        depth_img = np.array(depth_img).astype(np.float32)

        depth_max = np.max(depth_img)
        depth_img = depth_img / depth_max

        if self.opt.zoom_out_scale != 1 or rgb_img.shape[0] != depth_img.shape[0]:
            rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_AREA)
        rgb_img = rgb_img / 255.
        rgb_img = rgb_img.astype(np.float32)

        depth_img = np.expand_dims(depth_img, 2)
        rgbd_img = np.concatenate((depth_img, rgb_img), 2)

        rgbd_img = rgbd_img.transpose(2, 0, 1)
        return rgbd_img

    def loadDepthImg(self, data_path):
        data_path = data_path.decode('UTF-8')
        data_path_depth = data_path.split('|')[0]
        
        depth_img = cv2.imread(data_path_depth, 2)

        if self.opt.zoom_out_scale != 1:
            depth_img = cv2.resize(depth_img, (int(depth_img.shape[1] / self.opt.zoom_out_scale), int(depth_img.shape[0] / self.opt.zoom_out_scale)), interpolation=cv2.INTER_NEAREST)
        depth_img = np.array(depth_img).astype(np.float32)
        
        depth_max = np.max(depth_img)
        depth_img = depth_img / depth_max

        depth_img = np.expand_dims(depth_img, 2)
        depth_img = depth_img.transpose(2, 0, 1)
        return depth_img

    def __getitem__(self, index):
        data_item = {}
            
        if self.opt.use_rgb:
            data_item['A'] = self.loadRGBDImg(self.data_path_clean[index])
            if self.is_train:
                data_item['B'] = self.loadRGBDImg(self.data_path_noise[index])
        else:
            data_item['A'] = self.loadDepthImg(self.data_path_clean[index])
            if self.is_train:
                data_item['B'] = self.loadDepthImg(self.data_path_noise[index])
        
        data_item['A_paths'] = self.data_path_clean[index]
        if self.is_train:
            data_item['B_paths'] = self.data_path_noise[index]
        return data_item

    def getDataloader(self):
        if(self.is_train):
            return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        else:
            return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)

def preDataPathInterior(data_path, folder_list):
    all_file_list = []
    for i in range(len(folder_list)):
        file_list_depth = os.listdir(data_path + folder_list[i] + '/depth0/data')
        for j in range(len(file_list_depth)):
            file_rgb = data_path + folder_list[i] + '/cam0/data/' + file_list_depth[j]
            file_list_depth[j] = data_path + folder_list[i] + '/depth0/data/' + file_list_depth[j]
            paired_file_name = file_list_depth[j] + '|' + file_rgb

            all_file_list.append(paired_file_name)
    
    return all_file_list

def preDataPathScanNet(data_path, folder_list):
    all_file_list = []
    for i in range(len(folder_list)):
        file_list_depth = os.listdir(data_path + folder_list[i] + '/depth')
        for j in range(len(file_list_depth)):
            file_rgb = data_path + folder_list[i] + '/color/' + file_list_depth[j].split('.')[0] + '.jpg'
            file_list_depth[j] = data_path + folder_list[i] + '/depth/' + file_list_depth[j]
            paired_file_name = file_list_depth[j] + '|' + file_rgb
            all_file_list.append(paired_file_name)
    
    return all_file_list

def preDataPathImageTask(data_path, file_list):
    all_file_list = []
    for i in range(len(file_list)):
        file_path = data_path + file_list[i]
        all_file_list.append(file_path)

    return all_file_list

def saveH5(files_path, h5_path, target_name='datapath.h5'):
    files_path = np.array(files_path)

    with h5py.File(h5_path + target_name, 'w') as data_file:
        data_type = h5py.special_dtype(vlen=str)
        data = data_file.create_dataset('data_path', files_path.shape, dtype=data_type)
        data[:] = files_path
        data_file.close()

    print('Save path done!')

def datapathPrepare(opt):
    # initialize data path to dataPath.h5
    
    if opt.dataset == 'depthsynthesis':
        folder_list_clean = os.listdir(opt.data_path_clean)
        print('Number of clean models: ', len(folder_list_clean))

        files_path_clean = preDataPathInterior(opt.data_path_clean, folder_list_clean)
        print('Number of clean data: ', len(files_path_clean))
        saveH5(files_path_clean, opt.data_path_h5, 'DataPathClean.h5')
        
        folder_list_noise = os.listdir(opt.data_path_noise)
        print('Number of noise folders: ', len(folder_list_noise))

        files_path_noise = preDataPathScanNet(opt.data_path_noise, folder_list_noise)
        print('Number of noise data: ', len(files_path_noise))
        saveH5(files_path_noise, opt.data_path_h5, 'DataPathNoise.h5')
    else:
        file_list_A = os.listdir(opt.data_path_image + opt.dataset + '/trainA/')
        print('Number of A images: ', len(file_list_A))
        file_path_A = preDataPathImageTask(opt.data_path_image + opt.dataset + '/trainA/', file_list_A)
        saveH5(file_path_A, opt.data_path_h5, 'DataPath' + opt.dataset + 'A.h5')

        file_list_B = os.listdir(opt.data_path_image + opt.dataset + '/trainB/')
        print('Number of B images: ', len(file_list_B))
        file_path_B = preDataPathImageTask(opt.data_path_image + opt.dataset + '/trainB/', file_list_B)
        saveH5(file_path_B, opt.data_path_h5, 'DataPath' + opt.dataset + 'B.h5')
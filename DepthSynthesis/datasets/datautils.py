import os
import random

import numpy as np
import h5py
import cv2
from tqdm import tqdm

import torch
import torch.utils.data as data

from utils.utils import label_processing, lable_cropping, label_processing_cropping

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
        if type(data_path) == bytes:
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
        if type(data_path) == bytes:
            data_path = data_path.decode('UTF-8')
        data_path_depth = data_path.split('|')[0]
        data_path_rgb = data_path.split('|')[1]

        depth_img = cv2.imread(data_path_depth, 2)
        rgb_img = cv2.imread(data_path_rgb)

        if self.opt.rm_bkgd:
            data_path_label = data_path.split('|')[2]
            background_mask = label_processing(data_path_label)
            depth_img[background_mask] = 0

        if self.opt.is_crop:
            data_path_label = data_path.split('|')[2]
            label_id = int(data_path.split('|')[3])

            if 'LM' in self.opt.dataset:
                up, down, left, right = label_processing_cropping(data_path_label, label_id)
            else:
                up, down, left, right = lable_cropping(data_path_label, label_id)

            depth_img = depth_img[up:down, left:right]
            rgb_img = rgb_img[up:down, left:right]

        if self.opt.zoom_out_scale != 1:
            if self.opt.zoom_out_scale == 0:
                depth_img = cv2.resize(depth_img, (self.opt.fixed_size, self.opt.fixed_size), interpolation=cv2.INTER_NEAREST)
            else:
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
        if type(data_path) == bytes:
            data_path = data_path.decode('UTF-8')
        data_path_depth = data_path.split('|')[0]
        
        depth_img = cv2.imread(data_path_depth, 2)

        if self.opt.rm_bkgd:
            data_path_label = data_path.split('|')[2]
            background_mask = label_processing(data_path_label)
            depth_img[background_mask] = 0

        if self.opt.is_crop:
            data_path_label = data_path.split('|')[2]
            label_id = int(data_path.split('|')[3])

            if 'LM' in self.opt.dataset:
                up, down, left, right = label_processing_cropping(data_path_label, label_id)
            else:
                up, down, left, right = lable_cropping(data_path_label, label_id)

            depth_img = depth_img[up:down, left:right]

        if self.opt.zoom_out_scale != 1:
            if self.opt.zoom_out_scale == 0:
                depth_img = cv2.resize(depth_img, (self.opt.fixed_size, self.opt.fixed_size), interpolation=cv2.INTER_NEAREST)
            else:
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

def preDataPathKinectRealSense(data_path, folder_list, mode='/kinect', rm_bkgd=True, is_crop=True):
    all_file_list = []
    for i in range(len(folder_list)):
        if is_crop:
            obj_id_list_file = open(data_path + folder_list[i] + '/object_id_list.txt', 'r')
            obj_id_list = []
            for line in obj_id_list_file:
                obj_id_list.append(int(line))

        file_list_depth = os.listdir(data_path + folder_list[i] + mode +'/depth/')
        for j in range(len(file_list_depth)):
            file_rgb = data_path + folder_list[i] + mode + '/rgb/' + file_list_depth[j]
            file_depth = data_path + folder_list[i] + mode + '/depth/' + file_list_depth[j]
            if rm_bkgd or is_crop:
                file_label = data_path + folder_list[i] + mode + '/label/' + file_list_depth[j]
                paired_file_name = file_depth + '|' + file_rgb + '|' + file_label
            else:
                paired_file_name = file_depth + '|' + file_rgb

            if is_crop:
                for k in range(len(obj_id_list)):
                    all_file_list.append(paired_file_name + '|' + str(obj_id_list[k]))
            else:
                all_file_list.append(paired_file_name)
    
    return all_file_list

def preDataPathSynthetic(data_path, folder_list, mode='/kinect', rm_bkgd=True, is_crop=True, list_path=''):
    all_file_list = []
    for i in range(len(folder_list)):
        if is_crop:
            obj_id_list_file = open(list_path + folder_list[i] + '/object_id_list.txt', 'r')
            obj_id_list = []
            for line in obj_id_list_file:
                obj_id_list.append(int(line))

        file_list_depth = os.listdir(data_path + folder_list[i] + mode +'/synthetic_res/')
        for j in range(len(file_list_depth)):
            if 'depth' in file_list_depth[j]:
                file_rgb = data_path + folder_list[i] + mode + '/synthetic_res/' + file_list_depth[j].split('_')[0] + '_rgb.png'
                file_depth = data_path + folder_list[i] + mode + '/synthetic_res/' + file_list_depth[j]
                if rm_bkgd or is_crop:
                    file_label = data_path + folder_list[i] + mode + '/synthetic_res/' + file_list_depth[j].split('_')[0] + '_label.png'
                    paired_file_name = file_depth + '|' + file_rgb + '|' + file_label
                else:
                    paired_file_name = file_depth + '|' + file_rgb
                
                if is_crop:
                    for k in range(len(obj_id_list)):
                        all_file_list.append(paired_file_name + '|' + str(obj_id_list[k]))
                else:
                    all_file_list.append(paired_file_name)
    
    return all_file_list

def preDataPathLMReal(data_path, folder_list, rm_bkgd=True, is_crop=True):
    import yaml
    all_file_list = []
    for i in range(len(folder_list)):
        # if is_crop:
        #     label_info_path = data_path + folder_list[i] + '/gt.yml'
        #     with open(label_info_path, 'r') as label_info_yaml:
        #         label_info = yaml.safe_load(label_info_yaml)

        file_list_depth = os.listdir(data_path + folder_list[i] +'/depth/')
        for j in range(len(file_list_depth)):
            file_rgb = data_path + folder_list[i] + '/rgb/' + file_list_depth[j]
            file_depth = data_path + folder_list[i] + '/depth/' + file_list_depth[j]
            if rm_bkgd or is_crop:
                file_label = data_path + folder_list[i] + '/mask/' + file_list_depth[j]
                paired_file_name = file_depth + '|' + file_rgb + '|' + file_label
            else:
                paired_file_name = file_depth + '|' + file_rgb

            if is_crop:
                # for all objs
                # frame_id = int(file_list_depth[j].split('.')[0])
                # num_objs = len(label_info[frame_id])
                # for k in range(num_objs):
                #     mask_size = label_info[frame_id][k]['obj_bb']
                #     if mask_size[2] < 24 or mask_size[3] < 24:
                #         continue
                #     all_file_list.append(paired_file_name + '|' + str(label_info[frame_id][k]['obj_id']))
                
                # for single obj
                all_file_list.append(paired_file_name + '|' + str(int(folder_list[i])))
            else:
                all_file_list.append(paired_file_name)
    
    return all_file_list

def preDataPathLMSyntheic(data_path, folder_list, rm_bkgd=True, is_crop=True, list_path=''):
    import yaml
    all_file_list = []
    for i in range(len(folder_list)):
        # if is_crop:
        #     label_info_path = list_path + folder_list[i] + '/gt.yml'
        #     with open(label_info_path, 'r') as label_info_yaml:
        #         label_info = yaml.safe_load(label_info_yaml)

        file_list_depth = os.listdir(data_path + folder_list[i])
        for j in range(len(file_list_depth)):
            if 'depth' in file_list_depth[j]:
                file_rgb = data_path + folder_list[i] + '/' + file_list_depth[j].split('_')[0] + '_rgb.png'
                file_depth = data_path + folder_list[i] + '/' + file_list_depth[j]
                if rm_bkgd or is_crop:
                    file_label = data_path + folder_list[i] + '/' + file_list_depth[j].split('_')[0] + '_label.png'
                    paired_file_name = file_depth + '|' + file_rgb + '|' + file_label
                else:
                    paired_file_name = file_depth + '|' + file_rgb
                
                depth_temp = cv2.imread(file_depth, 2)
                up, down, left, right = label_processing_cropping(file_label, int(folder_list[i]))
                depth_temp = depth_temp[up:down, left:right]

                missing_mask = (depth_temp <= 0)
                missing_mask = missing_mask.astype(np.int8)
                all_missing = np.sum(np.sum(missing_mask, 1), 0)
                if (all_missing >= 8):
                    continue

                if is_crop:
                    # for all objs
                    # frame_id = int(file_list_depth[j].split('.')[0])
                    # num_objs = len(label_info[frame_id])
                    # for k in range(num_objs):
                    #     mask_size = label_info[frame_id][k]['obj_bb']
                    #     if mask_size[2] < 24 or mask_size[3] < 24:
                    #         continue
                    #     all_file_list.append(paired_file_name + '|' + str(label_info[frame_id][k]['obj_id']))

                    # for single obj
                    all_file_list.append(paired_file_name + '|' + str(int(folder_list[i])))
                else:
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
    
    # InterioNet2ScanNet
    if opt.dataset == 'IN2SNdepthsynthesis':
        folder_list_clean = os.listdir(opt.data_path_clean)
        print('Number of clean folders: ', len(folder_list_clean))

        files_path_clean = preDataPathInterior(opt.data_path_clean, folder_list_clean)
        print('Number of clean data: ', len(files_path_clean))
        saveH5(files_path_clean, opt.data_path_h5, 'DataPathClean.h5')
        
        folder_list_noise = os.listdir(opt.data_path_noise)
        print('Number of noise folders: ', len(folder_list_noise))

        files_path_noise = preDataPathScanNet(opt.data_path_noise, folder_list_noise)
        print('Number of noise data: ', len(files_path_noise))
        saveH5(files_path_noise, opt.data_path_h5, 'DataPathNoise.h5')
    # Synthetic2Kinect
    elif opt.dataset == 'S2Kdepthsynthesis':
        folder_list_clean = os.listdir(opt.data_path_clean)
        print('Number of clean folders: ', len(folder_list_clean))

        files_path_clean = preDataPathSynthetic(opt.data_path_clean, folder_list_clean, mode='/kinect', rm_bkgd=opt.rm_bkgd, is_crop=opt.is_crop, list_path=opt.data_path_noise)
        print('Number of clean data: ', len(files_path_clean))
        saveH5(files_path_clean, opt.data_path_h5, 'DataPathClean.h5')
        
        folder_list_noise = os.listdir(opt.data_path_noise)
        print('Number of noise folders: ', len(folder_list_noise))

        files_path_noise = preDataPathKinectRealSense(opt.data_path_noise, folder_list_noise, mode='/kinect', rm_bkgd=opt.rm_bkgd, is_crop=opt.is_crop)
        print('Number of noise data: ', len(files_path_noise))
        saveH5(files_path_noise, opt.data_path_h5, 'DataPathNoise.h5')
    # Synthetic2Realsense
    elif opt.dataset == 'S2Rdepthsynthesis':
        folder_list_clean = os.listdir(opt.data_path_clean)
        print('Number of clean folders: ', len(folder_list_clean))

        files_path_clean = preDataPathSynthetic(opt.data_path_clean, folder_list_clean, mode='/realsense', rm_bkgd=opt.rm_bkgd, is_crop=opt.is_crop, list_path=opt.data_path_noise)
        print('Number of clean data: ', len(files_path_clean))
        saveH5(files_path_clean, opt.data_path_h5, 'DataPathClean.h5')
        
        folder_list_noise = os.listdir(opt.data_path_noise)
        print('Number of noise folders: ', len(folder_list_noise))

        files_path_noise = preDataPathKinectRealSense(opt.data_path_noise, folder_list_noise, mode='/realsense', rm_bkgd=opt.rm_bkgd, is_crop=opt.is_crop)
        print('Number of noise data: ', len(files_path_noise))
        saveH5(files_path_noise, opt.data_path_h5, 'DataPathNoise.h5')
    # Synthetic2Real of LineMod
    elif opt.dataset == 'LMdepthsynthesis':
        folder_list_clean = os.listdir(opt.data_path_clean)
        print('Number of clean folders: ', len(folder_list_clean))

        files_path_clean = preDataPathLMSyntheic(opt.data_path_clean, folder_list_clean, rm_bkgd=opt.rm_bkgd, is_crop=opt.is_crop, list_path=opt.data_path_noise)
        print('Number of clean data: ', len(files_path_clean))
        saveH5(files_path_clean, opt.data_path_h5, 'DataPathClean.h5')
        
        folder_list_noise = os.listdir(opt.data_path_noise)
        print('Number of noise folders: ', len(folder_list_noise))

        files_path_noise = preDataPathLMReal(opt.data_path_noise, folder_list_noise, rm_bkgd=opt.rm_bkgd, is_crop=opt.is_crop)
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

def datafiltering(opt, path_file):
    data_paths_list = h5py.File(path_file, 'r')
    data_paths = np.array(data_paths_list['data_path'])

    filtered_data_path = []
    for i in tqdm(range(len(data_paths))):
        data_path = data_paths[i]
        if type(data_path) == bytes:
            data_path = data_path.decode('UTF-8')
        data_path_depth = data_path.split('|')[0]
        depth_img = cv2.imread(data_path_depth, 2)

        data_path_label = data_path.split('|')[2]
        label_id = int(data_path.split('|')[3])

        up, down, left, right = lable_cropping(data_path_label, label_id)
        depth_img = depth_img[up:down, left:right]
        if down - up >= 640 or down - up <= 64 or np.max(depth_img) <= 10:
            continue

        filtered_data_path.append(data_path)
    
    print('Number of filtered data: ', len(filtered_data_path))
    saveH5(filtered_data_path, opt.data_path_h5, 'DataPathFilteredClean.h5')

def datasampleshuffling(opt):
    clean_data_paths_list = h5py.File(opt.data_path_h5 + 'DataPathFilteredClean.h5', 'r')
    clean_data_paths = list(clean_data_paths_list['data_path'])
    random.shuffle(clean_data_paths)
    clean_data_paths = clean_data_paths[0:25600]
    saveH5(clean_data_paths, opt.data_path_h5, 'DataPathClean.h5')

    noise_data_paths_list = h5py.File(opt.data_path_h5 + 'DataPathFilteredNoise.h5', 'r')
    noise_data_paths = list(noise_data_paths_list['data_path'])
    random.shuffle(noise_data_paths)
    noise_data_paths = noise_data_paths[0:25600]
    saveH5(noise_data_paths, opt.data_path_h5, 'DataPathNoise.h5')
# Generate synthesised depth
import os
import math
import time
import datetime

import cv2
import h5py
import torch
import numpy as np

from datasets.datautils import DepthDataset, ImageTaskDataset, datapathPrepare

from options.test_options import TestOptions

from models import create_model

def test(opt, test_dataloader, num_test_batch, model):
    inference_time = 0.
    iter_data_time = time.time()
    
    times = []
    for i_test, data in enumerate(test_dataloader, 0):
        iter_start_time = time.time()
        t_data = iter_start_time - iter_data_time

        inference_start_time = time.time()

        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        
        if i_test == 0:
            model.setup(opt)    # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()

        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()   # calculate loss functions, get gradients, update network weights

        inference_time = time.time() - inference_start_time

        io_start_time = time.time()

        paths = model.get_image_paths()
        
        model.compute_visuals()
        vis_res = model.get_current_visuals()

        # A_inputs = vis_res['real_A']
        B_results = vis_res['fake_B']

        # A_inputs = A_inputs.permute(0, 2, 3, 1)
        # if 'depthsynthesis' in opt.dataset:
        #     A_inputs = A_inputs.detach().cpu().numpy()[:, :, :, 0]
        # else:
        #     A_inputs = A_inputs.detach().cpu().numpy()

        B_results = B_results.permute(0, 2, 3, 1)
        if 'depthsynthesis' in opt.dataset:
            B_results = B_results.detach().cpu().numpy()[:, :, :, 0]
        else:
            B_results = B_results.detach().cpu().numpy()
        B_results = np.clip(B_results, 0, 1)

        for i_batch in range(len(paths)):
            vis_path = paths[i_batch]
            if type(vis_path) == bytes:
                vis_path = vis_path.decode('UTF-8')

            B_result = B_results[i_batch]
            B_result = B_result * 255.

            if not 'depthsynthesis' in opt.dataset:
                B_result = cv2.cvtColor(B_result, cv2.COLOR_RGB2BGR)

            res_path = opt.test_dir + opt.dataset + '/res_' + vis_path.split('/')[-1].split('.')[0] + '_' + opt.model + '.png'

            if not os.path.exists(opt.test_dir + opt.dataset):
                os.makedirs(opt.test_dir + opt.dataset)
            cv2.imwrite(res_path, B_result)

        io_time = time.time() - io_start_time

        eta = (inference_time + t_data + io_time) * (num_test_batch - i_test - 1)
        eta = str(datetime.timedelta(seconds=int(eta)))

        print("Batch: %d/%d, ETA: %s (%.4fs inf. %.4fs load %.4fs write)" % (i_test, num_test_batch, eta, inference_time, t_data, io_time))

        iter_data_time = time.time()
    return 0

if __name__ == "__main__":
    opt = TestOptions().parse()

    if opt.data_path_prepared == False:
        datapathPrepare(opt)

    if 'depthsynthesis' in opt.dataset:
        data_path_A_list = h5py.File(opt.data_path_file_clean, 'r')
        data_path_A = np.array(data_path_A_list['data_path'])
    else:
        file_list_A = os.listdir(opt.data_path_image + opt.dataset + '/testA/')
        print('Number of test A images: ', len(file_list_A))

        data_path_A = []
        for i in range(len(file_list_A)):
            file_path = opt.data_path_image + opt.dataset + '/testA/' + file_list_A[i]
            data_path_A.append(file_path)

    if not os.path.exists(opt.test_dir):
        os.makedirs(opt.test_dir)

    if 'depthsynthesis' in opt.dataset:
        test_dataset = DepthDataset(opt, data_path_A)
    else:
        test_dataset = ImageTaskDataset(opt, data_path_A)

    num_test_batch = math.ceil(test_dataset.__len__() / opt.batch_size)

    test_dataloader = test_dataset.getDataloader()
    model = create_model(opt)
    
    test(opt, test_dataloader, num_test_batch, model)
import os
import math

import cv2
import torch
import numpy as np

from options.test_options import TestOptions

from models import create_model

def evaluation(opt, inputs_list, inputs_path, model):
    for i_test in range(len(inputs_list)):
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        
        if i_test == 0:
            model.setup(opt)    # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()

        data = {}
        data['A'] = inputs_list[i_test]
        data['A_paths'] = inputs_path[i_test]

        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()   # calculate loss functions, get gradients, update network weights

        paths = model.get_image_paths()
        
        model.compute_visuals()
        vis_res = model.get_current_visuals()

        B_results = vis_res['fake_B']

        B_results = B_results.permute(0, 2, 3, 1)
        if 'depthsynthesis' in opt.dataset:
            B_results = B_results.detach().cpu().numpy()[:, :, :, 0]
        else:
            B_results = B_results.detach().cpu().numpy()
        B_results = np.clip(B_results, 0, 1)

        vis_path = paths

        B_result = B_results[0]
        B_result = B_result * 255.

        if not 'depthsynthesis' in opt.dataset:
            B_result = cv2.cvtColor(B_result, cv2.COLOR_RGB2BGR)

        if not os.path.exists(opt.eval_res_dir):
            os.makedirs(opt.eval_res_dir)

        res_path = opt.eval_res_dir + vis_path
        cv2.imwrite(res_path, B_result)

def loadDepthImages(opt):
    files_list = os.listdir(opt.eval_dir)

    inputs_list = []
    inputs_path = []

    for i in range(len(files_list)):
        if 'rgb' in files_list[i]:
            continue
        else:
            # depth_input = cv2.imread(opt.eval_dir + files_list[i], 2)
            depth_input = cv2.imread(opt.eval_dir + files_list[i], cv2.IMREAD_GRAYSCALE)
            if opt.zoom_out_scale != 1:
                depth_input = cv2.resize(depth_input, (int(depth_input.shape[1] / opt.zoom_out_scale), int(depth_input.shape[0] / opt.zoom_out_scale)), interpolation=cv2.INTER_NEAREST)
            depth_input = np.array(depth_input).astype(np.float32)

            rgb_input = cv2.imread(opt.eval_dir + 'rgb_' + files_list[i])
            if opt.zoom_out_scale != 1 or rgb_input.shape[0] != rgb_input.shape[0]:
                rgb_input = cv2.resize(rgb_input, (depth_input.shape[1], depth_input.shape[0]), interpolation=cv2.INTER_AREA)

            depth_max = np.max(depth_input)
            depth_input = depth_input / depth_max

            rgb_input = rgb_input / 255.
            rgb_input = rgb_input.astype(np.float32)

            depth_input = np.expand_dims(depth_input, 2)
            rgbd_input = np.concatenate((depth_input, rgb_input), 2)

            rgbd_input = rgbd_input.transpose(2, 0, 1)
            rgbd_input = np.expand_dims(rgbd_input, 0)
            rgbd_input = torch.tensor(rgbd_input)
            inputs_list.append(rgbd_input)
            inputs_path.append(files_list[i])

    return inputs_list, inputs_path

def loadRGBImages(opt):
    files_list = os.listdir(opt.eval_dir)

    inputs_list = []
    inputs_path = []

    for i in range(len(files_list)):
        rgb_input = cv2.imread(opt.eval_dir + files_list[i])
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_BGR2RGB)

        if opt.zoom_out_scale != 1:
            if opt.zoom_out_scale == 0:
                rgb_input = cv2.resize(rgb_input, (opt.fixed_size, opt.fixed_size), interpolation=cv2.INTER_AREA)
            else:
                rgb_input = cv2.resize(rgb_input, (int(rgb_input.shape[1] / opt.zoom_out_scale), int(rgb_input.shape[0] / opt.zoom_out_scale)), interpolation=cv2.INTER_AREA)

        rgb_input = rgb_input / 255.
        rgb_input = rgb_input.astype(np.float32)

        rgb_input = rgb_input.transpose(2, 0, 1)
        rgb_input = np.expand_dims(rgb_input, 0)
        rgb_input = torch.tensor(rgb_input)
        inputs_list.append(rgb_input)
        inputs_path.append(files_list[i])

    return inputs_list, inputs_path

if __name__ == "__main__":
    opt = TestOptions().parse()

    if 'depthsynthesis' in opt.dataset:
        inputs_list, inputs_path = loadDepthImages(opt)
    else:
        inputs_list, inputs_path = loadRGBImages(opt)
    
    model = create_model(opt)
    evaluation(opt, inputs_list, inputs_path, model)
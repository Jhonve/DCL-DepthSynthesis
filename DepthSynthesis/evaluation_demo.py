import os
import math
import time
import datetime

import cv2
import torch
import numpy as np

from options.test_options import TestOptions

from models import create_model

def evaluation(opt, model):
    return 0

def loadDepthImages(opt):
    files_list = os.listdir(opt.eval_dir)

    inputs_list = []

    for i in range(files_list):
        if 'rgb' in files_list[i]:
            continue
        else:
            depth_input = cv2.imread(opt.eval_dir + files_list[i], cv2.IMREAD_GRAYSCALE)
            rgb_input = cv2.imread(opt.eval_dir + 'rgb_' + files_list[i])

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

    return inputs_list

def loadRGBImages(opt):
    files_list = os.listdir(opt.eval_dir)

    inputs_list = []

    for i in range(files_list):
        rgb_input = cv2.imread(opt.eval_dir + files_list[i])
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_BGR2RGB)
        rgb_input = rgb_input / 255.
        rgb_input = rgb_input.astype(np.float32)

        rgb_input = rgbd_input.transpose(2, 0, 1)
        rgb_input = np.expand_dims(rgb_input, 0)
        rgb_input = torch.tensor(rgb_input)
        inputs_list.append(rgb_input)

    return inputs_list

if __name__ == "__main__":
    opt = TestOptions().parse()

    if opt.dataset == 'depthsynthesis':
        inputs_list = loadDepthImages(opt)
    else:
        inputs_list = loadDepthImages(opt)
    
    model = create_model(opt)
    evaluation(model)
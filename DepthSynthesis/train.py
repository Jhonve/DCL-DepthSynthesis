import os
import time
import datetime
import random

import h5py
import torch
import numpy as np

from datasets.datautils import datapathPrepare
from datasets.datautils import DepthDataset, ImageTaskDataset

from options.train_options import TrainOptions

from models import create_model

from tensorboardX import SummaryWriter

def train(opt, train_dataloader, val_dataloader, num_train_batch, model):
    tsboard_writer = SummaryWriter('runs/' + opt.name)

    total_iters = 0
    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration

        for i_train, data in enumerate(train_dataloader, 0):
            iter_start_time = time.time()   # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += 1
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            if total_iters % opt.print_freq == 0:
                optimize_start_time = time.time()

            if epoch == opt.epoch_count and i_train == 0:
                if opt.model == 'cut' or opt.model == 'dcl':
                    model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            if total_iters % opt.print_freq == 0:
                # optimize_time = (time.time() - optimize_start_time) / opt.batch_size * 0.005 + 0.995 * optimize_time
                optimize_time = time.time() - optimize_start_time

            if total_iters % opt.display_freq == 0:
                model.compute_visuals()
                vis_imgs = model.get_current_visuals()
                for img_label, img_value in vis_imgs.items():
                    if opt.use_rgb == True and opt.dataset == 'depthsynthesis':
                        vis_img = img_value[:, 0:1, :, :].detach().cpu().numpy()
                    else:
                        vis_img = img_value.detach().cpu().numpy()
                    vis_img = np.clip(vis_img, 0, 1)
                    tsboard_writer.add_images(img_label, vis_img, global_step=(epoch - 1) * num_train_batch + i_train + 1)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                for loss_label, loss_value in losses.items():
                    tsboard_writer.add_scalar(loss_label, loss_value, global_step=(epoch - 1) * num_train_batch + i_train + 1)
                
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            if total_iters % opt.print_freq == 0:
                iter_data_time = time.time()

                eta = (optimize_time + t_data) * num_train_batch * (opt.n_epochs + opt.n_epochs_decay - epoch) +\
                    (optimize_time + t_data) * (num_train_batch - i_train - 1)
                eta = str(datetime.timedelta(seconds=int(eta)))

                print("Epoch: %d/%d; Batch: %d/%d, ETA: %s (%.4fs opt. %.4fs load)" % 
                        (epoch, opt.n_epochs + opt.n_epochs_decay, i_train + 1, num_train_batch, 
                            eta, optimize_time, t_data))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    return 0

# split validation dataset

def splitData(num_val_batch, batch_size, data_path_A, data_path_B, validation_file):
    if data_path_A.shape[0] < data_path_B.shape[0]:
        num_data = data_path_A.shape[0]
    else:
        num_data = data_path_B.shape[0]

    num_val_data = num_val_batch * batch_size
    num_train_data = num_data - num_val_data

    val_index = random.sample(range(0, num_data), num_val_data)
    train_index = list(set(range(0, num_data)) - set(val_index))

    num_train_batch = int(len(train_index) / batch_size)

    train_path_A = data_path_A[train_index]
    train_path_B = data_path_B[train_index]

    val_path_A = data_path_A[val_index]
    val_path_B = data_path_B[val_index]

    val_index = np.array(val_index)
    np.save(validation_file, val_index)

    return train_path_A, train_path_B, val_path_A, val_path_B, num_train_batch

def resplitData(batch_size, data_path_A, data_path_B, validation_file):
    if data_path_A.shape[0] < data_path_B.shape[0]:
        num_data = data_path_A.shape[0]
    else:
        num_data = data_path_B.shape[0]

    val_index = np.load(validation_file)
    val_index = list(val_index)
    train_index = list(set(range(0, num_data)) - set(val_index))

    num_train_batch = int(len(train_index) / batch_size)

    train_path_A = data_path_A[train_index]
    train_path_B = data_path_B[train_index]

    val_path_A = data_path_A[val_index]
    val_path_B = data_path_B[val_index]

    return train_path_A, train_path_B, val_path_A, val_path_B, num_train_batch

if __name__ == '__main__':
    opt = TrainOptions().parse()

    if opt.data_path_prepared == False:
        datapathPrepare(opt)

    if not os.path.exists(opt.validation_dir):
        os.makedirs(opt.validation_dir)
    
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
        os.makedirs(opt.checkpoints_dir + '/' + opt.name)

    if not os.path.exists(opt.checkpoints_dir + '/' + opt.name):
        os.makedirs(opt.checkpoints_dir + '/' + opt.name)
    
    if 'depthsynthesis' in opt.dataset:
        data_path_A_list = h5py.File(opt.data_path_file_clean, 'r')
        data_path_B_list = h5py.File(opt.data_path_file_noise, 'r')
        data_path_A = np.array(data_path_A_list['data_path'])
        data_path_B = np.array(data_path_B_list['data_path'])
    else:
        data_path_A_list = h5py.File(opt.data_path_file_image + opt.dataset + 'A.h5', 'r')
        data_path_B_list = h5py.File(opt.data_path_file_image + opt.dataset + 'B.h5', 'r')
        data_path_A = np.array(data_path_A_list['data_path'])
        data_path_B = np.array(data_path_B_list['data_path'])

    validation_file = opt.validation_dir + opt.splited_index_file

    if opt.epoch_count > 1 or os.path.exists(opt.validation_dir + opt.splited_index_file):
        train_path_A, train_path_B, val_path_A, val_path_B, num_train_batch = \
            resplitData(opt.batch_size, data_path_A, data_path_B, validation_file)
    else:
        train_path_A, train_path_B, val_path_A, val_path_B, num_train_batch = \
            splitData(opt.num_val_batch, opt.batch_size, data_path_A, data_path_B, validation_file)

    # initialize Dataloader
    if 'depthsynthesis' in opt.dataset:
        train_dataset = DepthDataset(opt, train_path_A, train_path_B)
    else:
        train_dataset = ImageTaskDataset(opt, train_path_A, train_path_B)
    train_dataloader = train_dataset.getDataloader()

    if opt.num_val_batch > 0:
        if 'depthsynthesis' in opt.dataset:
            val_dataset = DepthDataset(opt, val_path_A, val_path_B)
        else:
            val_dataset = ImageTaskDataset(opt, val_path_A, val_path_B)
        val_dataloader = val_dataset.getDataloader()
    else:
        val_dataloader = None
    
    # initialize Network structure etc.
    Network_model = create_model(opt)

    # start train
    train(opt, train_dataloader, val_dataloader, num_train_batch, Network_model)
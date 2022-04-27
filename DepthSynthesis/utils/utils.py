import os
import random
import argparse

import cv2
import numpy as np
import torch

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def label_processing(label_path):
    label = cv2.imread(label_path)
    label_max = np.max(label)

    if label_max == 255: # is rendered synthetic label
        label = label[:, :, 0]
        background_mask = (label >= 90) # there are 88 models in total, reduce rasterization error
    else:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        background_mask = (label <= 0)

    return background_mask

def lable_cropping(label_path, label_id, bias=8):
    label_img = cv2.imread(label_path)
    h = label_img.shape[0]
    w = label_img.shape[1]

    idx_h, idx_w, _ = np.where(label_img == label_id + 1)

    if idx_h.shape[0] == 0: # no such label
        up = int((h / 2) - 128)
        down = int((h / 2) + 128)
        left = int((w / 2) - 128)
        right = int((w / 2) + 128)
        return up, down, left, right

    up = np.min(idx_h) - bias
    down = np.max(idx_h) + bias
    left = np.min(idx_w) - bias
    right = np.max(idx_w) + bias

    if right - left >= down - up:
        mid_h = (up + down) / 2
        len = right - left
        up = int(mid_h - (len / 2))
        down = int(mid_h + (len / 2))

        # check bbox
        if up < 0:
            up = 0
            down = up + len
        elif down >= h:
            down = h - 1
            up = down - len

        if left < 0:
            left = 0
            right = left + len
        elif right >= w:
            right = w - 1
            left = right - len
    else:
        mid_w = (left + right) / 2
        len = down - up
        left = int(mid_w - (len / 2))
        right = int(mid_w + (len / 2))

        # check bbox
        if up < 0:
            up = 0
            down = up + len
        elif down >= h:
            down = h - 1
            up = down - len

        if left < 0:
            left = 0
            right = left + len
        elif right >= w:
            right = w - 1
            left = right - len

    return up, down, left, right

def label_processing_cropping(label_path, label_id, bias=8):
    label_img = cv2.imread(label_path)
    h = label_img.shape[0]
    w = label_img.shape[1]

    bg_h, bh_w, _ = np.where(label_img == 0)
    if bg_h.shape[0] >= 16:
        # is real mask
        up, down, left, right = lable_cropping(label_path, label_id=254)
    else:
        up, down, left, right = lable_cropping(label_path, label_id * 3 - 1)

    return up, down, left, right
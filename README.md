# DCL-DepthSynthesis (DCL)
Unpaired Depth Synthesis Using Differential Contrastive Learning 

We provide our PyTorch implementaion of paper 'Differential Contrastive Learning for Geometry-Aware Depth Synthesis'

## Prerequisites

- Linux (Ubuntu is suggested)
- Python 3
- NVIDIA GPU, CUDA and CuDNN

## Requirements

- pytorch >= 1.4
- tensorboardX
- numpy, h5py, opencv-python

## Usage

### Evaluation for depth synthesis



### Training for depth synthesis

1. Download [InteriorNet](https://interiornet.org/) and [ScanNet](http://www.scan-net.org/) datasets.

2. Extract depth and rgb frames from two datasets into different folders.

3. Start training

   ```
   cd DepthSynthesis
   
   python train.py --dataset depthsynthesis --data_path_clean your_path_to_interiornet --data_path_noise your_path_to_scannet
   ```

### Training for RGB image translation

1. Download datasets follow [this](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/datasets/download_cut_dataset.sh).

2. Start training

   ```
   cd DepthSynthesis
   
   python train.py --dataset RGB_dataset(for example: horse2zebra) --data_path_image your_path_to_RGB_datasets --input_nc 3 --output_nc 3
   ```

## Acknowledgments

Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation). We also thank [Synchronized-BatchNorm-PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for synchronized batchnorm implementation.
import argparse
import os
from utils import utils
import torch
import models

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='depthsynthesis', help='name of the experiment. It decides where to store samples and models')
        # parser.add_argument('--name', type=str, default='DebugTest', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='dcl', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normG', type=str, default='batch', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='batch', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        parser.add_argument('--is_sync_norm', type=bool, default=True, help='whether to use normalization on multi-gpu')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=utils.str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
        parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        # dataset parameters
        parser.add_argument('--dataset', type=str, default='depthsynthesis', choices=['depthsynthesis', 'horse2zebra', 'apple2orange', 'summer2winter_yosemite', 'monet2photo'], help='choose whihc dataset to train.')
        parser.add_argument('--data_path_prepared', type=bool, default=False, help='Wherether to generate depth path file.')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
        parser.add_argument('--data_path_clean', type=str, default='../../Dataset/InteriorNet/random/', help='Path to InteriorNet dataset')
        parser.add_argument('--data_path_noise', type=str, default='../../Dataset/ScanNet/', help='Path to ScanNet dataset')
        parser.add_argument('--data_path_image', type=str, default='../../Dataset/ImageTasksDatasets/', help='Path to image task datasets')
        parser.add_argument('--data_path_h5', type=str, default='./datasets/', help='Path to h5 file for saving data pathes')
        parser.add_argument('--zoom_out_scale', type=int, default=2, help='Doing scaling in the training step, using 0 for image tasks')
        parser.add_argument('--fixed_size', type=int, default=256, help='The fixed size in the training step, when setting \'zoom_out_scale == 0\'')
        parser.add_argument('--data_path_file_clean', type=str, default='./datasets/DataPathClean.h5', help='H5 file saved clean data path')
        parser.add_argument('--data_path_file_noise', type=str, default='./datasets/DataPathNoise.h5', help='H5 file saved noise data path')
        parser.add_argument('--data_path_file_image', type=str, default='./datasets/DataPath', help='H5 file saved noise data path')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # netwoek parameters for depth synthesis
        parser.add_argument('--use_rgb', type=bool, default=True, help='if specified, print more debugging information')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

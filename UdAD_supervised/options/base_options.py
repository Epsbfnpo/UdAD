import argparse
import os
from utils.utils import *
import torch
import models
import data


class BaseOptions:
    """
    This class defines and manages the command-line options for the experiment.

    It handles the initialization, modification, and parsing of command-line arguments used to configure various aspects of the experiment, such as model architecture, dataset handling, and training parameters.
    """

    def __init__(self):
        """
        Initializes the BaseOptions class.

        This constructor sets up the basic state of the class, primarily to track whether the options have been initialized.
        """
        self.initialized = False

    def initialize(self, parser):
        """
        Defines the basic command-line arguments for the experiment.

        This method sets up the fundamental arguments that are common across different experiments, including paths, model configurations, and dataset parameters.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the options are added.

        Returns:
            argparse.ArgumentParser: The argument parser with the basic options added.
        """
        # Basic parameters
        parser.add_argument('--dataroot', required=True, help='Path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment. Determines where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs: e.g. 0  0,1,2, 0,2. Use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='/home/sheng/Muscle_Seg_updated/checkpoints', help='Directory to save models')
        # Model and network basic parameters
        parser.add_argument('--model', type=str, default='MuscleSeg', help='Chooses which model to use. [ u | more_to_implement ]')
        parser.add_argument('--net', type=str, default='U2D', help='Specify net architecture [resu | fe | mc | u | more_to_implement ]')
        parser.add_argument('--input_nc', type=int, default=1, help='Number of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='Number of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--cnum', type=int, default=16, help='Intermediate feature channels')
        parser.add_argument('--init_type', type=str, default='normal', help='Network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='Scaling factor for normal, xavier, and orthogonal initializations')
        # Dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='mixed', help='Chooses how datasets are loaded. [thigh | femur | mixed | more_to_implement ]')
        parser.add_argument('--serial_batches', action='store_true', help='If true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='Input batch size')
        parser.add_argument('--input_patch_size', default=8, type=int, help='Actual number of input patches')
        parser.add_argument('--data_norm', type=str, default='instance_norm_3D', help='Chooses how to normalize the data')
        # Additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='Which epoch to load? Set to "latest" to use the latest cached model')
        parser.add_argument('--load_iter', type=int, default=0, help='Which iteration to load? If load_iter > 0, load models by iter_[load_iter]; otherwise, load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='If specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='Customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True
        return parser

    def gather_options(self):
        """
        Gathers and parses the command-line options.

        This method initializes the parser if it hasn't been initialized, adds model-specific and dataset-specific options, and finally parses the command-line arguments.

        Returns:
            argparse.Namespace: The parsed command-line options.
        """
        if not self.initialized:  # Check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Get the basic options
        opt, _ = parser.parse_known_args()

        # Modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # Parse again with new defaults

        # Modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # Save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """
        Prints and saves the command-line options.

        This method prints the current configuration of options to the console and saves them to a text file in the experiment's directory.

        Args:
            opt (argparse.Namespace): The parsed command-line options.
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

        # Save to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """
        Parses the command-line options and performs additional processing.

        This method gathers and parses the command-line options, applies any suffix specified in the options, and sets up the GPU IDs.

        Returns:
            argparse.Namespace: The processed command-line options.
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # Train or test

        # Process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # Set GPU IDs
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

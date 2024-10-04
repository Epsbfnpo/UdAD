import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler

class BaseModel(ABC):
    """
    Abstract base class for all models.

    This class defines common functionalities and utilities that can be used by any derived model class, including saving/loading networks, updating learning rates, and setting up optimizers and schedulers.
    """

    def __init__(self, opt):
        """
        Initializes the BaseModel with given options.

        Args:
            opt (object): Configuration options for the model, including GPU settings, checkpoint directory, and training flags.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # Set device to GPU or CPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # Directory to save checkpoints
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # Used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Adds or modifies command-line options specific to the model.

        This method can be overridden by subclasses to add or modify command-line options.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
            is_train (bool): Flag indicating whether training or testing mode.

        Returns:
            argparse.ArgumentParser: The modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """
        Abstract method to set the input data for the model.

        This method must be implemented in subclasses to define how the input data is set for the model.

        Args:
            input (dict): The input data for the model.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Abstract method for the forward pass of the model.

        This method must be implemented in subclasses to define the forward pass through the model.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Abstract method to optimize the model's parameters.

        This method must be implemented in subclasses to define how the model's parameters are optimized during training.
        """
        pass

    def setup(self, opt):
        """
        Sets up the model, including initializing schedulers and loading networks.

        Args:
            opt (object): Configuration options for the model.
        """
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def eval(self):
        """
        Sets the model to evaluation mode.

        This disables dropout and batch normalization layers from updating their parameters, which is important during inference.
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def test(self):
        """
        Performs a forward pass in evaluation mode without calculating gradients.

        Returns:
            torch.Tensor: The output of the model after the forward pass.
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

        with torch.no_grad():
            output = self.forward()
            return output

    def get_image_paths(self):
        """
        Returns the paths of the input images.

        This is useful for keeping track of which images were processed.

        Returns:
            list: A list of image paths.
        """
        return self.image_paths

    def update_learning_rate(self):
        """
        Updates the learning rate based on the scheduler's policy.

        This method checks the learning rate policy and updates the learning rate accordingly.
        """
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """
        Returns a dictionary of the current losses.

        This method is useful for logging and monitoring training progress.

        Returns:
            OrderedDict: A dictionary where the keys are loss names and the values are the current loss values.
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # Convert to float for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """
        Saves the model networks to disk.

        Args:
            epoch (int): The current epoch number, used to name the checkpoint files.
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                print('save model to path:', save_path)
                net = getattr(self, 'net_' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """
        Fixes issues with InstanceNorm checkpoints prior to PyTorch 0.4.

        This method removes keys related to running_mean, running_var, and num_batches_tracked from the state_dict if they are set to None.

        Args:
            state_dict (dict): The state dictionary containing model parameters.
            module (torch.nn.Module): The module to be patched.
            keys (list of str): The keys pointing to the parameter or buffer in the state_dict.
            i (int): The current index in the keys list.
        """
        key = keys[i]
        if i + 1 == len(keys):  # At the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, load_path=None):
        """
        Loads the model networks from disk.

        Args:
            epoch (int): The epoch number of the checkpoint to load.
            load_path (str, optional): The path to the checkpoint file. If None, it is constructed from the save directory.
        """
        for name in self.model_names:
            if isinstance(name, str):
                if load_path is not None:
                    load_path = load_path
                else:
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)

                state_dict = torch.load(load_path, map_location=str(self.device))

                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # Patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # Need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def load_net(self, net, load_path):  # Load single network added by sheng
        """
        Loads a single network from a checkpoint file.

        Args:
            net (torch.nn.Module): The network to load.
            load_path (str): The path to the checkpoint file.
        """
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        print('loading the model from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(self.device))

        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # Patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # Need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """
        Prints the details of all networks, including the total number of parameters.

        Args:
            verbose (bool): If True, prints the full architecture of each network.
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = sum(param.numel() for param in net.parameters())
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Sets the requires_grad attribute for all networks, controlling whether their parameters are updated during backpropagation.

        Args:
            nets (list or torch.nn.Module): The network or list of networks to modify.
            requires_grad (bool): If False, the gradients are not calculated for the networks.
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_scheduler(self, optimizer, opt):
        """
        Returns a learning rate scheduler based on the specified policy.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to which the scheduler is applied.
            opt (object): Configuration options containing the learning rate policy.

        Returns:
            torch.optim.lr_scheduler: The learning rate scheduler.
        """
        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def set_current_epoch(self, epoch):
        """
        Sets the current epoch number, useful for resuming training.

        Args:
            epoch (int): The current epoch number.
        """
        self.current_epoch = epoch

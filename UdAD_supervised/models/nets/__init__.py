import torch
import torch.nn as nn
from torch.nn import init
import importlib


class Identity(nn.Module):
    """
    A placeholder identity layer that returns the input unchanged.

    This layer can be used as a default or placeholder in neural networks when no operation is required.
    """
    def forward(self, x):
        """
        Forward pass that returns the input as is.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The same input tensor.
        """
        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initializes the weights of a neural network.

    This function applies a specified type of weight initialization to all layers of the network that contain learnable weights, such as Conv and Linear layers.

    Args:
        net (torch.nn.Module): The neural network whose weights need to be initialized.
        init_type (str): The type of initialization to apply. Options are 'normal', 'xavier', 'kaiming', and 'orthogonal'.
        init_gain (float): Scaling factor for the initialization.

    Raises:
        NotImplementedError: If an unsupported initialization type is provided.
    """
    print(f"<models/nets/__init__.py> - Initializing weights with type '{init_type}' and gain {init_gain}.")

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initializes a neural network and prepares it for training.

    This function moves the network to the specified GPU(s) and applies the desired weight initialization.

    Args:
        net (torch.nn.Module): The neural network to be initialized.
        init_type (str): The type of weight initialization to apply.
        init_gain (float): Scaling factor for the initialization.
        gpu_ids (list of int): List of GPU IDs to which the network should be moved.

    Returns:
        torch.nn.Module: The initialized neural network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def find_net_using_name(net_name):
    """
    Finds and returns the neural network class using the provided name.

    This function dynamically imports a module corresponding to the given network name and searches for a class within that module that matches the expected class name (formed by removing underscores from the network name and appending 'net').

    Args:
        net_name (str): The name of the network module to be imported and the class to be searched for.

    Returns:
        class: The network class that matches the given name.

    Raises:
        SystemExit: If no matching class is found, the program exits with an error message.
    """
    net_filename = "models.nets." + net_name + "_net"
    netlib = importlib.import_module(net_filename)
    net = None
    target_net_name = net_name.replace('_', '') + 'net'
    for name, cls in netlib.__dict__.items():
        if name.lower() == target_net_name.lower():
            net = cls

    if net is None:
        print("In %s.py, there should be a net with class name that matches %s in lowercase." % (net_filename, target_net_name))
        exit(0)

    return net


def create_net(opt):
    """
    Creates and initializes a neural network based on the provided options.

    This function finds the appropriate network class using the network name specified in the options, instantiates it, and initializes its weights.

    Args:
        opt (object): Configuration options containing network type, initialization type, initialization gain, and GPU IDs.

    Returns:
        torch.nn.Module: The initialized neural network ready for training or inference.
    """
    net_class = find_net_using_name(opt.net)
    net = net_class(opt)

    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)

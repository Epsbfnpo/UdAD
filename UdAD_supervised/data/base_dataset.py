import torch.utils.data as data
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """
    Abstract base class for datasets.

    This class serves as a template for creating datasets compatible with PyTorch's
    data loading utilities. It inherits from both `torch.utils.data.Dataset` and
    `ABC` (Abstract Base Class), enforcing the implementation of key methods
    like `__len__` and `__getitem__` in any subclass.

    Attributes:
        opt (object): Configuration options passed during initialization.
        root (str): The root directory for the dataset, typically defined in `opt`.
    """

    def __init__(self, opt):
        """
        Initializes the BaseDataset class with the provided options.

        Args:
            opt (object): An object containing various configuration options,
                          including the data root directory.
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Adds or modifies command-line options specific to the dataset.

        This method is used to dynamically modify command-line options for training
        or testing. It can be overridden in subclasses to introduce additional
        dataset-specific options.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which options are added.
            is_train (bool): A flag indicating whether the mode is training or testing.

        Returns:
            argparse.ArgumentParser: The modified argument parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        This method must be implemented in any subclass of BaseDataset. It defines
        the size of the dataset and is used by PyTorch's DataLoader to determine
        the length of an epoch.

        Returns:
            int: The total number of samples in the dataset.
        """
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        Retrieves a data sample for the given index.

        This method must be implemented in any subclass of BaseDataset. It defines
        how individual data samples are accessed and returned, typically as a
        dictionary or tuple containing both data and labels.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            object: The data sample corresponding to the given index.
        """
        pass

from abc import ABC, abstractmethod


class BaseIO(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.isTrain = opt.isTrain

    @abstractmethod
    def load_sample(self, index):
        pass

    @abstractmethod
    def save_sample(self, sample):
        pass

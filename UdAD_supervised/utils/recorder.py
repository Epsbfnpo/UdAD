import os
import time
import warnings
from utils.utils import *
from tensorboardX import SummaryWriter


class Recorder:
    """
    This class handles recording and logging of training progress.

    It manages the creation of logs for training losses and writes them to both a file and TensorBoard for visualization. It also provides methods for printing and saving the training losses at each epoch.
    """

    def __init__(self, opt):
        """
        Initializes the Recorder class.

        This constructor sets up the logging directory, creates a log file for storing training losses, and initializes a TensorBoard writer for visualizing the losses.

        Args:
            opt (object): Configuration options containing paths for checkpoints and other settings.
        """
        self.opt = opt  # Cache the options
        log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard')
        mkdirs(log_dir)

        # Create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=opt.name, flush_secs=20)

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def plot_current_losses(self, current_iters, losses):
        """
        Plots the current losses to TensorBoard.

        This method writes the current losses to TensorBoard at each iteration, allowing for real-time visualization of training progress.

        Args:
            current_iters (int): The current iteration number.
            losses (dict): A dictionary of loss names and their corresponding values.
        """
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v, current_iters)
            warnings.simplefilter(action='ignore', category=FutureWarning)
        self.writer.flush()

    def print_epoch_losses(self, epoch, losses, time):
        """
        Prints and logs the losses for the current epoch.

        This method prints the loss values for the current epoch to the console and appends them to the loss log file.

        Args:
            epoch (int): The current epoch number.
            losses (dict): A dictionary of loss names and their corresponding values.
            time (float): The time taken for the epoch, in seconds.
        """
        message = '(epoch: %d, time: %.2f mins) ' % (epoch, time / 60)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def close(self):
        """
        Closes the TensorBoard writer.

        This method ensures that all pending events are flushed to TensorBoard and then closes the writer, finalizing the logs.
        """
        self.writer.close()
        warnings.simplefilter(action='ignore', category=FutureWarning)

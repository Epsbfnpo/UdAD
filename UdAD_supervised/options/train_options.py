from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class defines options for the training phase of the experiment.

    It extends the BaseOptions class and adds specific command-line options related to training, such as checkpoint saving, learning rate scheduling, and optimizer parameters.
    """

    def initialize(self, parser):
        """
        Initializes and defines the training-specific command-line options.

        This method extends the basic options provided by the BaseOptions class and adds additional options specific to the training phase, including parameters for saving checkpoints, managing learning rates, and configuring the optimizer.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the options are added.

        Returns:
            argparse.ArgumentParser: The argument parser with training-specific options added.
        """
        parser = BaseOptions.initialize(self, parser)  # Define shared options from BaseOptions

        # Network saving and loading parameters
        parser.add_argument('--phase', type=str, default='train', help='Specify phase: train, val, test, preprocess, etc.')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='Frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='Frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='Continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='Starting epoch count; models are saved by <epoch_count>, <epoch_count>+<save_latest_freq>, etc.')

        # Training parameters
        parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='Number of epochs to linearly decay the learning rate to zero')

        # Optimizer parameters
        parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of the Adam optimizer')
        parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='linear', help='Learning rate policy: [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='Multiply learning rate by a gamma every lr_decay_iters iterations')

        self.isTrain = True  # Indicate that this is the training phase

        return parser

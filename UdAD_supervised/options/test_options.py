from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    This class defines options for the testing phase of the experiment.

    It extends the BaseOptions class and adds specific command-line options related to testing, such as the phase type, results directory, and evaluation mode.
    """

    def initialize(self, parser):
        """
        Initializes and defines the testing-specific command-line options.

        This method extends the basic options provided by the BaseOptions class and adds additional options specific to the testing phase.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the options are added.

        Returns:
            argparse.ArgumentParser: The argument parser with testing-specific options added.
        """
        parser = BaseOptions.initialize(self, parser)  # Define shared options from BaseOptions
        parser.add_argument('--phase', type=str, default='test', help='Specify phase: train, val, test, etc.')
        parser.add_argument('--results_dir', type=str, default='/home/sheng/Muscle_Seg_updated/results', help='Directory to save test results')
        parser.add_argument('--save_prediction', type=int, default=4, help='Frequency of saving predicted images')

        # Dropout and BatchNorm have different behaviors during training and testing.
        parser.add_argument('--eval', action='store_true', help='Use evaluation mode during testing')
        parser.add_argument('--num_test', type=int, default=20, help='Number of test images to process')

        # Set default values specifically for testing
        parser.set_defaults(model='test')

        self.isTrain = False  # Indicate that this is not a training phase

        return parser

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_false', help='use eval mode during test time.')

        parser.add_argument('--eval_dir', type=str, default='./datasets/eval_depth/', help='path to the evaluation datasets')
        parser.add_argument('--eval_res_dir', type=str, default='./datasets/eval_outputs/', help='where to save evaluation results')

        parser.add_argument('--is_write', type=bool, default=True, help='whether to write results')
        parser.add_argument('--test_dir', type=str, default="../../TestRes/ImageTasks/", help='where to save results')

        self.isTrain = False
        return parser
from .base_options import BaseOptions


def nullable_string(val):
    if not val:
        return None
    return val


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--fod_path', type=str, default='./dataset/fodf/test_hcpa', help="The path to the fod image")

        parser.add_argument('--output_path', type=str, default='./dataset/NFO_prediction/hcpa', help="The output path")

        parser.add_argument('--weights_path', type=str, default='./checkpoints/NFO_hcp/500.pth')

        self.isTrain = False
        return parser

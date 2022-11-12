from argparse import ArgumentParser

parser = ArgumentParser(description='This is a test program')
parser.add_argument('--backbone', type=str, default='resnet18', help='backbone')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--resize', type=int, default=224, help='resize')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
# parser.add_argument('--sample_ratio', type=float, default=None, help='sample ratio')
parser.add_argument('--ovs', action='store_true', help='oversampling')
parser.add_argument('--aug', action='store_true', help='augmentation')
args = parser.parse_args()
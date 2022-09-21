from argparse import ArgumentParser

parser = ArgumentParser(description='This is a test program')
parser.add_argument('--backbone', type=str, default='resnet18', help='backbone')
parser.add_argument('--input_channel', type=int, default=3, help='input channel')
parser.add_argument('--pretrained', type=eval, help='pretrained')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--save_path', type=str, default='model.pth', help='save path')
args = parser.parse_args()
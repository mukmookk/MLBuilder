import argparse

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='', metavar='PATH', help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--val_size', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--model', type=str, default=None, metavar='MODEL', help='model name (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
parser.add_argument('--buffer_size', type=int, default=2000, help='buffer size for shuffling (default: 2000)')

args = parser.parse_args()

class Config:
    def __init__(self):
        self.image_dir = args.dir
        self.datset = args.dataset
        self.image_size = args.data_path
        self.batch_size = args.batch_size
        self.val_size = args.val_size
        self.model = args.model
        self.epochs = args.epochs
        self.lr_init = args.lr_init
        self.decay = args.wd
        self.momentum = args.momentum
        self.buffer_size = args.buffer_size
import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
optimizer_names = ["sgd", "adam", "rmsprop"]

def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser(description='Digit Recognition')
    parser.add_argument('--dataset-root', default='./data',
                        help='train dataset path')
    parser.add_argument('--arch', default='mobilenetv2_cifar', choices=model_names,
                        help='model architecture: {} (default: mobilenetv2_cifar)'.format(' | '.join(model_names)))
    parser.add_argument('--gpu-id', type=int, default=-1,
                        help='gpu called when train')
    parser.add_argument('--alphabet', default='0123456789',
                        help='label alphabet, string format or file')
    parser.add_argument('--optimizer', default='rmsprop', choices=optimizer_names,
                        help='optimizer options: {} (default: rmsprop)'.format(' | '.join(optimizer_names)))
    parser.add_argument('--max-epoch', type=int, default='30',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                        help='initialize model with random weights (default: pretrained on cifar10)')
    parser.add_argument('--validate-interval', type=int, default=1,
                        help='Interval to be displayed')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='save a model')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size to train a model')
    parser.add_argument('--train-samples', default=640000, type=int,
                        help='train sample number')
    parser.add_argument('--image-size', type=int, default=32,
                        help='maximum size of longer image side used for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency (default: 10)')
    parser.add_argument('--directory', metavar='EXPORT_DIR', default='./checkpoint',
                        help='Where to store samples and models')
    parser.add_argument('--rnn', action='store_true',
                        help='Train the model with model of rnn')
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')
    parser.add_argument('--test-only', action='store_true',
                        help='test only')
    args = parser.parse_args()
    return args
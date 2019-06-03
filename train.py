import os
import time
import math
import random
import shutil
import contextlib
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.converter import LabelConverter, IndexConverter
from datasets.dataset import InMemoryDigitsDataset, DigitsDataset, collate_train, collate_dev, inmemory_train, inmemory_dev
from generate import gen_text_img

import arguments
from models.densenet_ import DenseNet


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images = sample.images
        images = images.to(device)
        targets = sample.targets
        targets = targets.to(device)
        target_lengths = sample.target_lengths

        optimizer.zero_grad()

        log_probs = model(images)
        input_lengths = torch.full((images.size(0),),log_probs.size(0), dtype=torch.int32, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        losses.update(loss.item())
        loss.backward()

        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg


def validate(dev_loader, model, epoch, converter):
    batch_time = AverageMeter()
    accuracy = AverageMeter()

    model.eval()

    num_correct = 0
    num_verified = 0
    end = time.time()
    
    for i, sample in enumerate(dev_loader):
        images = sample.images
        images = images.to(device)
        targets = sample.targets

        optimizer.zero_grad()

        log_probs = model(images)
        preds = converter.best_path_decode(log_probs, strings=False)
        
        batch_time.update(time.time() - end)
        end = time.time()

        for i in range(len(targets)):
            num_verified += len(targets[i])
        for pred, target in zip(preds, targets):
            if(pred == target):
                num_correct += 1
        accuracy.update(num_correct / num_verified) # character-level
        
#         for i in range(len(preds)): #打印pred的结果
#             pred = converter.best_path_decode(log_probs, strings=True)[i].decode('utf-8')
#             print('pred: {}'.format(pred))
            
        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(dev_loader):
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Accu {accuracy.val:.3f}'.format(
                   epoch+1, i+1, len(dev_loader), batch_time=batch_time, accuracy=accuracy))

    return accuracy.val

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, '{}_epoch_{}.pth.tar'.format(state['arch'], state['epoch']))
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)
    torch.save(state, filename)
    if is_best:
        print('>>>> save best model at epoch: {}'.format(state['epoch']))
        filename_best = os.path.join(directory, '{}_best.pth.tar'.format(state['arch']))
        with contextlib.suppress(FileNotFoundError):
            os.remove(filename_best)
        shutil.copyfile(filename, filename_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
if __name__=="__main__":
    import sys
    sys.argv = ['main.py','--dataset-root','alphabet','--alphabet','alphabet/alphabet_decode_5990.txt',
            '--lr','5e-5','--max-epoch','200','--gpu-id','-1','--not-pretrained']  

    args = arguments.parse_args()
    
    device = torch.device("cpu")
    
    if os.path.isfile(args.alphabet):
        alphabet = ''
        with open(args.alphabet, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                alphabet += line.strip()
        args.alphabet = alphabet

    num = 200 # 喂的图片个数
    dev_num = num
    use_file = 1
    text = "嘤嘤嘤"
    text_length = 1
    font_size = -1
    font_id = -1
    space_width = 1
    text_color = '#282828'
    thread_count = 8
    channel = 3

    random_skew = True
    skew_angle = 2
    random_blur = True
    blur = 0.5

    orientation = 0
    distorsion = -1
    distorsion_orientation = 2
    background = 1

    random_process = True
    noise = 20
    erode = 2
    dilate = 2
    incline = 10

    iteration = 1 #iteration的个数=喂几组不一样的图片

    transform = transforms.Compose([
        transforms.Resize((32, 280)),
        transforms.ToTensor(),
    ])

    model = DenseNet(num_classes=len(args.alphabet) + 1)
    
    if args.pretrained:
        model_path = 'pretrained/new_prarams2.pth'
        checkpoint = torch.load(model_path,map_location = 'cpu')
        model.load_state_dict(checkpoint)

    criterion = nn.CTCLoss()
    # critierion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = optim.SGD(params=model.parameters(),lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    label_converter = LabelConverter(args.alphabet, ignore_case=False)

    is_best = False
    best_accuracy = 0.0
    accuracy = 0.0
    start_epoch = 0

    acc = []

    for i in range(iteration):
        text_meta, text_img = gen_text_img(num, use_file, text, text_length, font_size, font_id, space_width,
                                           background, text_color,
                                           orientation, blur, random_blur, distorsion, distorsion_orientation,
                                           skew_angle, random_skew,
                                           random_process, noise, erode, dilate, incline,
                                           thread_count)
        dev_meta, dev_img = text_meta, text_img

        index_converter = IndexConverter(args.alphabet, ignore_case=True)

        train_dataset = InMemoryDigitsDataset(mode='train', text=text_meta, img=text_img, total=num,
                                             transform=transform, converter = index_converter)
        dev_dataset = InMemoryDigitsDataset(mode='dev', text=dev_meta, img=dev_img, total=num,
                                           transform=transform, converter = index_converter)

        train_loader = data.DataLoader(dataset=train_dataset,batch_size=args.batch_size, num_workers=4, shuffle=True,
                                       collate_fn=collate_train, pin_memory=True)
        dev_loader = data.DataLoader(dataset=dev_dataset,batch_size=args.batch_size, num_workers=4, shuffle=False,
                                     collate_fn=collate_dev, pin_memory=False)

        for epoch in range(start_epoch, args.max_epoch):
            loss = train(train_loader, model, criterion, optimizer, epoch)

            if (epoch + 1) % args.validate_interval == 0:
                with torch.no_grad():
                    accuracy = validate(dev_loader, model, epoch, label_converter)

            is_best = accuracy > 0.0 and accuracy >= best_accuracy
            best_accuracy = max(accuracy, best_accuracy)
            acc.append(format(accuracy))
            print('>>>> Accuracy: {}'.format(accuracy))

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint({
                    'arch': args.arch,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy,


                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.directory)

            if best_accuracy == 1:
                break
                
    torch.save(model.state_dict(), 'pretrained/new_prarams.pth')
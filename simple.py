import os
import time
import math
import random
import shutil
import contextlib
import matplotlib.pyplot as plt
%matplotlib inline

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
    model.train()
    for i, sample in enumerate(train_loader):
        images = sample.images
        targets = sample.targets
        target_lengths = sample.target_lengths

        optimizer.zero_grad()

        log_probs = model(images)
        input_lengths = torch.full((images.size(0),),log_probs.size(0), dtype=torch.int32, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()

        optimizer.step()
    return loss


def validate(dev_loader, model, epoch, converter):
    model.eval()
    accuracy = 0.0
    for i, sample in enumerate(dev_loader):
        images = sample.images
        targets = sample.targets
        target_lengths = sample.target_lengths

        optimizer.zero_grad()

        log_probs = model(images)
        preds = converter.best_path_decode(log_probs, strings=False)

        num_correct = 0
        num_verified = 0
        for i in range(len(targets)):
            num_verified += len(targets[i])
        for pred, target in zip(preds, targets):
            if(pred == target):
                num_correct += 1
        accuracy = num_correct/num_verified

    return accuracy


if __name__ == '__main__':

    os.system("python main.py --dataset-root alphabet --arch densenet121 --alphabet alphabet/alphabet_decode_5990.txt --batch-size 64 --lr 5e-5 --max-epoch 100 --gpu-id -1 --not-pretrained")
    args = arguments.parse_args()

    device = torch.device("cpu")

    if os.path.isfile(args.alphabet):
        alphabet = ''
        with open(args.alphabet, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                alphabet += line.strip()
        args.alphabet = alphabet

    num = 100
    dev_num = num
    use_file = 0
    text = "嘤嘤嘤"
    text_length = 10
    font_size = 32
    font_id = 1
    space_width = 1
    text_color = '#282828'
    thread_count = 8
    channel = 3

    random_skew = True
    skew_angle = 0
    random_blur = True
    blur = 0

    orientation = 0
    distorsion = 0
    distorsion_orientation = 0
    background = 1

    random_process = False
    noise = 0
    erode = 0
    dilate = 0
    incline = 0

    model = DenseNet(num_classes=len(args.alphabet) + 1)

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

    round = 5
    for i in range(round):
        text_meta, text_img = gen_text_img(num, use_file, text, text_length, font_size, font_id, space_width,
                                           background, text_color,
                                           orientation, blur, random_blur, distorsion, distorsion_orientation,
                                           skew_angle, random_skew,
                                           random_process, noise, erode, dilate, incline,
                                           thread_count)
        dev_meta, dev_img = text_meta, text_img

        index_converter = IndexConverter(args.alphabet, ignore_case=True)

        train_dataset = InMemoryDigitsDataset(mode='train', text=text_meta, img=text_img, total=num)
        dev_dataset = InMemoryDigitsDataset(mode='dev', text=dev_meta, img=dev_img, total=num)

        train_loader = data.DataLoader(dataset=train_dataset, num_workers=4, shuffle=True,
                                       collate_fn=inmemory_train, pin_memory=True)
        dev_loader = data.DataLoader(dataset=train_dataset, num_workers=4, shuffle=False,
                                     collate_fn=inmemory_dev, pin_memory=False)

        for epoch in range(start_epoch, args.max_epoch):
            loss = train(model, model, criterion, optimizer, epoch)

            if (epoch + 1) % args.validate_interval == 0:
                with torch.no_grad():
                    accuracy = validate(dev_loader, model, epoch, label_converter)

            is_best = accuracy > 0.0 and accuracy >= best_accuracy
            best_accuracy = max(accuracy, best_accuracy)
            acc.append(format(accuracy))
            print('>>>> Accuracy: {}'.format(accuracy))

            if best_accuracy == 1:
                break


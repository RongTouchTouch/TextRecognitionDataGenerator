import os
import glob

import torch
import torch.utils.data as data
from datasets.datahelpers import default_loader


class InMemoryDigitsDataset(data.Dataset):
    """ Digits dataset."""
    def __init__(self, mode, data_root, transform=None, loader=default_loader):
        # self.img_names = glob.glob('{}/*.jpg'.format(label_path))
        if not (mode == 'train' or mode == 'dev'):
            raise(RuntimeError("MODE should be either train or dev, passed as string"))

        self.mode = mode
        self.transform = transform
        self.loader = loader
        self.img_root = os.path.join(data_root, 'images')

        self.img_names = []
        self.targets = []

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_root, self.img_names[idx])
        image = self.loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[idx]
        target = torch.IntTensor([int(i) for i in target])

        return image, target


class DigitsDataset(data.Dataset):
    """ Digits dataset."""
    def __init__(self, mode, data_root, transform=None, loader=default_loader):
        # self.img_names = glob.glob('{}/*.jpg'.format(label_path))
        if not (mode == 'train' or mode == 'dev'):
            raise(RuntimeError("MODE should be either train or dev, passed as string"))

        self.mode = mode
        self.transform = transform
        self.loader = loader
        self.img_root = os.path.join(data_root, 'images')

        self.img_names = []
        self.targets = []

        label_path = os.path.join(data_root, '{}.txt'.format(mode))
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                self.img_names.append(line[0])
                self.targets.append(line[1:])
                # print(self.img_names)
                # print(self.targets)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_root, self.img_names[idx])
        image = self.loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[idx]
        target = torch.IntTensor([int(i) for i in target])

        return image, target


class DigitsBatchTrain:
    def __init__(self, batch):
        transposed_data = list(zip(*batch))
        self.images = torch.stack(transposed_data[0], 0)
        self.targets = torch.cat(transposed_data[1], 0)
        self.target_lengths = torch.IntTensor([len(i) for i in transposed_data[1]])

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.targets = self.targets.pin_memory()
        self.target_lengths = self.target_lengths.pin_memory()
        return self.images, self.targets, self.target_lengths


def collate_train(batch):
    return DigitsBatchTrain(batch)


class DigitsBatchDev:
    def __init__(self, batch):
        transposed_data = list(zip(*batch))
        self.images = torch.stack(transposed_data[0], 0)
        self.targets = [i.tolist() for i in transposed_data[1]]

    def pin_memory(self):
        self.images = self.images.pin_memory()
        return self.images, self.targets


def collate_dev(batch):
    return DigitsBatchDev(batch)

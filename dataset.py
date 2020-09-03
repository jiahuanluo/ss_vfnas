# encoding: utf-8

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random


class MultiViewDataset:

    def __init__(self, data_dir, data_type, height, width):
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.89156885, 0.89156885, 0.89156885],
                                 std=[0.18063523, 0.18063523, 0.18063523]),
        ])

        self.classes, self.class_to_idx = self.find_class(data_dir)
        subfixes = ['_' + str(((i - 1) * 30)).zfill(3) + '_' + str(i).zfill(3) + '.png' for i in range(1, 13)]
        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            all_off_files = ['_'.join(item.split('_')[:-2]) for item in all_files]
            all_off_files = sorted(list(set(all_off_files)))

            for single_off_file in all_off_files:
                all_views = [single_off_file + sg_subfix for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]
                for i in range(6):
                    sample = [all_views[i], all_views[i + 6]]
                    self.x.append(sample)
                    self.y.append([self.class_to_idx[label]])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):  # this is single_indexx
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(2):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        labels.append(self.y[indexx])

        return np.array(data[0]), np.array(data[1]), np.array(labels).ravel()


class MultiViewDataset6Party:

    def __init__(self, data_dir, data_type, height, width, k):
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.89156885, 0.89156885, 0.89156885],
                                 std=[0.18063523, 0.18063523, 0.18063523]),
        ])

        self.classes, self.class_to_idx = self.find_class(data_dir)
        # subfixes = ['_' + str(((i - 1) * 30)).zfill(3) + '_' + str(i).zfill(3) + '.png' for i in range(1, 13)]
        subfixes = ['_' + str(i).zfill(3) + '.png' for i in range(1, 13)]
        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            # all_off_files = ['_'.join(item.split('_')[:-2]) for item in all_files]
            all_off_files = [item.split('.')[0] for item in all_files if item[-3:] == 'off']
            all_off_files = sorted(list(set(all_off_files)))

            for single_off_file in all_off_files:
                all_views = [single_off_file + sg_subfix for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]
                # if data_type == "test":
                #     random.shuffle(all_views)
                # for i in range(13):
                # sample = [all_views[j + i * 6] for j in range(0, k)]
                sample = [all_views[j] for j in range(0, k)]
                self.x.append(sample)
                self.y.append([self.class_to_idx[label]])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):  # this is single_indexx
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


def test_dataset():
    DATA_DIR = './data/modelnet40v2png/'
    train_dataset = MultiViewDataset6Party(DATA_DIR, 'train', 32, 32, 1)
    valid_dataset = MultiViewDataset6Party(DATA_DIR, 'test', 32, 32, 1)
    n_train = len(train_dataset)
    n_valid = len(valid_dataset)
    print(n_train)
    print(n_valid)
    train_indices = list(range(n_train))
    valid_indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               sampler=train_sampler,
                                               num_workers=2,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=2,
                                               sampler=valid_sampler,
                                               num_workers=2,
                                               pin_memory=True)
    for i, (x1, y) in enumerate(train_loader):
        print(x1[0].shape, y.shape)
        break


if __name__ == "__main__":
    test_dataset()

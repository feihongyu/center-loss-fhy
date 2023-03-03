import os
from math import ceil, floor

import cv2
from torch.utils import data


def create_datasets(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)

    images_root = dataroot

    names = os.listdir(images_root)
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    training_set = []
    validation_set = []
    for klass, name in enumerate(names):
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)

        images = os.listdir(os.path.join(images_root, name))
        total = len(images)

        training_set += map(
                add_class,
                images[:ceil(total * train_val_split)])
        validation_set += map(
                add_class,
                images[floor(total * train_val_split):])

    return training_set, validation_set, len(names)


class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = cv2.imread((self.datasets[index][0]))
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])

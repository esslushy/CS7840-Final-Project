import os
import cv2
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image
import torch
from tqdm import tqdm

def load_mnist_font_dataset(train_transforms, test_transforms):
    X = []
    Y = []
    for i in range(10):
        for d in os.listdir("print_digit/dataset/assets/{}".format(i)):
            t_img = cv2.imread("print_digit/dataset/assets/{}".format(i)+"/"+d)
            t_img = cv2.cvtColor(t_img,cv2.COLOR_BGR2GRAY)
            X.append(t_img)
            Y.append(i)

    X = np.array(X, dtype=np.float32) / 255.
    Y = np.array(Y)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]

    mnist_train = PrintedDigitDataset(X, Y, mode='train', transform=train_transforms)
    mnist_test = PrintedDigitDataset(X, Y, mode='test', transform=test_transforms)
    return mnist_train, mnist_test

class PrintedDigitDataset(Dataset):
    def __init__(self, X, Y, mode, transform=None):
        assert mode in ['train', 'test', 'valid']
        self.transform = transform
        if mode == 'train':
            self.data = X[:5000].astype(np.float32)
            self.targets = Y[:5000].astype(np.int64)
        else:
            self.data = X[5000:6000].astype(np.float32)
            self.targets = Y[5000:6000].astype(np.int64)
        self.num_samples = len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

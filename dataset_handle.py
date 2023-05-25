import os
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class fMRI_digits_Dataset(Dataset):
    def __init__(self, data, labels, list):    
        self.data = data
        self.labels = labels
        self.list = list

    def __getitem__(self, idx):
        return self.data[self.list[idx]], self.labels[self.list[idx]]

    def __len__(self):
        return len(self.list)
    
def get_fMRI_digits_dataset(BATCH_SIZE = 5):
    fmri = torch.load(os.path.abspath(os.path.dirname(__file__)) + '/data/digits-fmri').astype(np.float32)
    imgs = torch.load(os.path.abspath(os.path.dirname(__file__)) + '/data/digits-images').astype(np.float32)

    train_fmri = np.concatenate([fmri[0:45], fmri[50:95]])
    test_fmri = np.concatenate([fmri[45:50], fmri[95:100]])

    train_imgs = np.expand_dims(np.concatenate([imgs[0:45], imgs[50:95]]), 1)/255.
    test_imgs = np.expand_dims(np.concatenate([imgs[45:50], imgs[95:100]]), 1)/255.

    train_fmri_dataset = fMRI_digits_Dataset(train_fmri, train_imgs, range(train_imgs.shape[0]))
    test_fmri_dataset = fMRI_digits_Dataset(test_fmri, test_imgs, range(test_imgs.shape[0]))
    train_fmri_loader = DataLoader(dataset = train_fmri_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_fmri_loader = DataLoader(dataset = test_fmri_dataset, batch_size = 10, shuffle = False)
    return train_fmri_loader, test_fmri_loader

def get_mnist_dataset(mnist_path = './data/', BATCH_SIZE = 256):
    train_mnist_dataset = datasets.MNIST(root = mnist_path,
                                train = True,
                                transform = transforms.Compose([transforms.ToTensor()]),
                                download = True)

    train_mnist_loader = torch.utils.data.DataLoader(train_mnist_dataset,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)

    test_mnist_dataset = datasets.MNIST(root = mnist_path,
                                train = False,
                                transform = transforms.Compose([transforms.ToTensor()]),
                                download = True)

    test_mnist_loader = torch.utils.data.DataLoader(test_mnist_dataset,
                                                batch_size = BATCH_SIZE,
                                                shuffle = False)
    return train_mnist_loader, test_mnist_loader
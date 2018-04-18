from torch.utils.data import Dataset
import PIL.Image as im
from scipy import misc
import numpy as np
from torchvision import transforms
import torch


class coco224Dataset(Dataset):
    def __init__(self, data_path, labels_path, ids_file_path):
        ids_file = open(ids_file_path)
        self.ids = ids_file.read().split("\n")[:-1]

        self.data_path = data_path
        self.labels_path = labels_path

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = im.open(self.data_path+id+'.jpg')
        label = np.load(self.labels_path+id+'.npz')['arr_0']

        img = self.toTensor(img)
        img = self.normalize(img)

        return {'image':img, 'label':label}


class voc224Dataset(Dataset):
    def __init__(self, data_path, labels_path, ids_file_path):
        ids_file = open(ids_file_path)
        self.ids = ids_file.read().split("\n")[:-1]

        self.data_path = data_path
        self.labels_path = labels_path

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = im.open(self.data_path+id+'.jpg')
        label = np.asarray(im.open(self.labels_path+id+'.png'))
        img = self.toTensor(img)
        img = self.normalize(img)

        return {'image':img, 'label':label}
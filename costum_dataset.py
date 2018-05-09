from torch.utils.data import Dataset
import PIL.Image as im
import numpy as np
from torchvision import transforms


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


class VOCDataset(Dataset):
    def __init__(self, data_path, labels_path, class_labels_path, ids_file_path, img_h=224, img_w=224, evaluate=False):
        ids_file = open(ids_file_path)
        self.ids = ids_file.read().split("\n")[:-1]

        self.data_path = data_path
        self.labels_path = labels_path
        self.class_labels_path = class_labels_path
        self.h = img_h
        self.w = img_w
        self.evaluate = evaluate

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = im.open(self.data_path+id+'.jpg')
        label = im.open(self.labels_path+id+'.png')
        class_label = im.open(self.class_labels_path+id+'.png')
        size = label.size

        if self.evaluate:
            img, label, class_label = resize_sample(img, label, class_label, self.h, self.w)

        label = np.asarray(label)
        class_label = np.asarray(class_label)

        img = self.toTensor(img)
        img = self.normalize(img)
        return {'image':img, 'label':label, 'class_label':class_label, 'size':size}


def resize_sample(img, label, class_label, h, w, restore=False):
    '''
    utility function to resize sample(PIL image and label) to a given dimension
    without cropping information. the network takes in tensors with dimensions
    that are multiples of 32.
    :param img: PIL image to resize
    :param label: PIL image with the label to resize
    :param h: desired height
    :param w: desired width
    :param restore: set this to true when you want to restore a padded image to it's
                    original dimensions
    :return: the resized image, label
    '''
    center_crop = transforms.CenterCrop([h,w])

    old_size = img.size  # old_size is in (width, height) format
    w_ratio = float(w) / old_size[0]
    h_ratio = float(h) / old_size[1]
    if restore:
        ratio = max(w_ratio, h_ratio)
    else:
        ratio = min(w_ratio, h_ratio)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, im.ANTIALIAS)
    label = label.resize(new_size, im.ANTIALIAS)

    img = center_crop(img)
    label = center_crop(label)

    if class_label is not None:
        class_label = class_label.resize(new_size, im.ANTIALIAS)
        class_label.center_crop(class_label)

    return img, label, class_label



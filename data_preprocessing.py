from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pycocotools import mask as maskUtils
import PIL.Image as Image
import PIL
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from config import *

coco = COCO(train_annotations)


#def main():
    #preprocess_data(train_images_path, train_annotations, processed_train2017_images, processed_train2017_labels)


def preprocess_data(image_path, anns_path, image_destination, labels_destination):
    i = 0
    dataloader = get_dataloader(image_path, anns_path)
    for sampled_batch in dataloader:
        i += 1
        print(i)
        for item in sampled_batch:
            image = item[0]
            anns = item[1]
            if anns==[]:
                continue
            image_id = anns[0]['image_id']
            seg = annsToSeg(anns)
            np.save(labels_destination + str(image_id) +'.npy', seg)
            image.save(image_destination + str(image_id) +'.jpg')


def annsToSeg(anns):
    '''
    converts COCO-format annotations of a given image to a PASCAL-VOC style label
     !!!No guarantees where segmentations overlap - might lead to loss of objects!!!
     resizes label to (224, 224)
    :param anns: COCO annotations as return by 'coco.loadAnns'
    :return: a 2D numpy array (of type int32) where the value of each pixel is the ID of the instance
                to which it belongs
    '''
    image_details = coco.loadImgs(anns[0]['image_id'])[0]

    h = image_details['height']
    w = image_details['width']
    short_side = min(h, w)
    rescale_ratio = 224.0/short_side

    seg = np.zeros((h, w))
    masks, anns = annsToMask(anns, h, w)

    for i,mask in enumerate(masks):
        seg = np.where(seg>0,seg, mask*anns[i]['id'])

    seg = seg.astype(np.int32)
    seg = ndimage.zoom(seg, rescale_ratio, order=0, prefilter=False)
    seg = crop_center(seg, 224,224)
    return seg



def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annsToMask(anns, h, w):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.
    :return: a list of binary masks (each a numpy 2D array) of all the annotations in anns
    """
    masks = []
    anns = sorted(anns, key=lambda x: x['area'])  # Smaller items first, so they are not covered by overlapping segs
    for ann in anns:
        rle = annToRLE(ann, h, w)
        m = maskUtils.decode(rle)
        masks.append(m)
    return masks, anns


def collate_fn(batch):
    return batch


def get_dataloader(data_path, anns_path, batch_size):

    image_transforms = transforms.Compose(
                            [transforms.Scale(224, Image.BILINEAR),
                             transforms.CenterCrop(224)])
                            #transforms.ToTensor())
    #label_transforms = transforms.Compose(
                            #[transforms.Lambda(annsToSeg)])

    # Prepare datasets, dataloaders
    coco_validation_dataset = datasets.CocoDetection(data_path, anns_path, image_transforms)
    coco_validation_dataloader = DataLoader(coco_validation_dataset, batch_size, collate_fn=collate_fn)

    return coco_validation_dataloader


def print_inter(im):
    print(im.size)
    print(im)
    plt.imshow(im)
    plt.show()
    return im


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

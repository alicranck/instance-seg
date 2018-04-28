import logging
import os
import torch
import numpy as np
from feature_extractor import *

embedding_dim = 32

float_type = torch.FloatTensor
double_type = torch.DoubleTensor
int_type = torch.IntTensor
long_type = torch.LongTensor

# Data root
data_root = 'E:\\Almog\\DLProjectData\\coco\\'

# Local paths to images, annotations (COCO dataset)
train_images_path = 'E:\\Almog\\DLProjectData\\coco\\train2017'
validation_images_path = 'E:\Almog\DLProjectData\coco\\val2017'
train_annotations = 'E:\Almog\DLProjectData\coco\\annotations_trainval2017\\annotations\\instances_train2017.json'
val_annotations = 'E:\Almog\DLProjectData\coco\\annotations_trainval2017\\annotations\\instances_val2017.json'

# Processed Images+Labels folders COCO
processed_val_root = 'E:\\Almog\\DLProjectData\\coco\\processed_val2017\\'
processed_train_root = 'E:\\Almog\\DLProjectData\\coco\\processed_train2017\\'

# Processed Images+Labels folders VOC
voc_processed_images = 'E:\Almog\DLProjectData\VOC2012\processedImages\\'
voc_processed_labels = 'E:\Almog\DLProjectData\VOC2012\processedLabels\\'

voc_train_ids = 'E:\Almog\DLProjectData\VOC2012\ImageSets\Segmentation\\train.txt'
voc_val_ids = 'E:\Almog\DLProjectData\VOC2012\ImageSets\Segmentation\\val.txt'


# Checkpoints and logs directory
chkpts_dir = 'C:\\Almog\\2018a\\DLProject\\instance_segmentation\\FeatureExtractor_checkpoints\\'


def config_experiment(name, resume=True):

    exp = {}
    os.makedirs(chkpts_dir+name, exist_ok=True)
    logger = config_logger(name)

    if resume:

        try:
            exp = torch.load(chkpts_dir+name+'\\chkpt.pth')
            logger.info("loading checkpoint, experiment: " + name)
            return exp, logger
        except:
            logger.warning('checkpoint does not exist. creating new experiment')

    model = FeatureExtractor()
    exp['model_state_dict'] = model.state_dict()
    exp['epoch'] = 0
    exp['best_loss'] = None
    exp['train_loss'] = []
    exp['val_loss'] = []

    return exp, logger


def save_experiment(exp, name, isBest=False):
    torch.save(exp,chkpts_dir+name+'\\chkpt.pth')
    if isBest:
        torch.save(exp, chkpts_dir + name + '\\best.pth')


def config_logger(current_exp):
    logger = logging.getLogger(current_exp)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler2 = logging.FileHandler(chkpts_dir+current_exp+'\\log')
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(handler2)

    return logger
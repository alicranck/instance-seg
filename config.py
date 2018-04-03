import logging
import os
import torch
import numpy as np
from feature_extractor import *

embedding_dim = 64

float_type = torch.FloatTensor
double_type = torch.DoubleTensor
int_type = torch.IntTensor

# Data root
data_root = 'E:\\Almog\\DLProjectData\\coco\\'

# Local paths to images, annotations (COCO dataset)
train_images_path = 'E:\\Almog\\DLProjectData\\coco\\train2017'
validation_images_path = 'E:\Almog\DLProjectData\coco\\val2017'
train_annotations = 'E:\Almog\DLProjectData\coco\\annotations_trainval2017\\annotations\\instances_train2017.json'
val_annotations = 'E:\Almog\DLProjectData\coco\\annotations_trainval2017\\annotations\\instances_val2017.json'

# Processed Images+Labels folders
processed_val_root = 'E:\\Almog\\DLProjectData\\coco\\processed_val2017\\'
processed_train_root = 'E:\\Almog\\DLProjectData\\coco\\processed_train2017\\'

# Checkpoints and logs directory
chkpts_dir = 'C:\\Almog\\2018a\\DLProject\\instance_segmentation\\FeatureExtractor_checkpoints\\'


def config_experiment(name, resume=True, lr=0.001):

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
    exp['optimizer_state_dict'] = torch.optim.Adam(model.parameters(), lr).state_dict()
    exp['epoch'] = 0
    exp['best_loss'] = None
    exp['loss_history'] = []

    return exp, logger


def save_experiment(exp, name, isBest=False):
    torch.save(exp,chkpts_dir+name+'\\chkpt.pth')
    if isBest:
        torch.save(exp, chkpts_dir + name + '\\best.pth')


def config_logger(current_exp):
    logger = logging.getLogger()
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
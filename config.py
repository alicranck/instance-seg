import logging
import os
from model import *

# Hyper parameters
embedding_dim = 32
batch_size = 32
learning_rate = 0.0003
lr_decay = 0.98
max_epoch_num = 100
context = True

# Checkpoints and logs directory - make sure to set local paths
chkpts_dir = '/model_checkpoints/'

if torch.cuda.is_available():
    float_type = torch.cuda.FloatTensor
    double_type = torch.cuda.DoubleTensor
    int_type = torch.cuda.IntTensor
    long_type = torch.cuda.LongTensor
else:
    float_type = torch.FloatTensor
    double_type = torch.DoubleTensor
    int_type = torch.IntTensor
    long_type = torch.LongTensor


def config_experiment(name, resume=True, context=context):

    exp = {}
    os.makedirs(chkpts_dir+name, exist_ok=True)
    logger = config_logger(name)

    if resume:

        try:
            exp = torch.load(chkpts_dir+name+'/chkpt.pth',map_location=lambda storage, loc: storage)
            logger.info("loading checkpoint, experiment: " + name)
            return exp, logger
        except Exception as e:
            logger.warning('checkpoint does not exist. creating new experiment')

    fe = FeatureExtractor(embedding_dim, context=context)
    exp['fe_state_dict'] = fe.state_dict()
    exp['epoch'] = 0
    exp['best_dice'] = None
    exp['train_fe_loss'] = []
    exp['val_fe_loss'] = []
    exp['dice_history'] = []

    return exp, logger


def save_experiment(exp, name, isBest=False):
    torch.save(exp,chkpts_dir+name+'/chkpt.pth')
    if isBest:
        torch.save(exp, chkpts_dir + name + '/best.pth')


def config_logger(current_exp):
    logger = logging.getLogger(current_exp)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler2 = logging.FileHandler(chkpts_dir+current_exp+'/log')
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(handler2)

    return logger
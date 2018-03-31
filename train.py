from torch.autograd import Variable
import torch.autograd
from torchvision import datasets
from visualization import *
from config import *
from costum_dataset import *
from torch.utils.data import DataLoader
from loss import CostumeLoss, loss
from logger import Logger
import logging

VISUALIZE = False

if torch.cuda.is_available():
    float_type = torch.cuda.FloatTensor
    double_type = torch.cuda.DoubleTensor
    int_type = torch.cuda.IntTensor


def printgradnorm(self, grad_input, grad_output):
    if (grad_input[0]!=grad_input[0]).data.any():
        print ("grad input nan")
    if (grad_output[0]!=grad_output[0]).data.any():
        print("grad output nan")
    if (grad_output[0]!=grad_output[0]).data.any() or (grad_input[0]!=grad_input[0]).data.any():
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].data.norm())
        print('grad_output norm:', grad_output[0].data.norm())
        return

# Hyper parameters
k = 10
batch_size = 16
learning_rate = 0.001
max_epoch_num = 1
current_experiment = 'test02'
current_data_root = processed_train_root

# Paths to data, labels
data_path = current_data_root + 'images\\'
labels_path = current_data_root + 'labels\\'
ids_path = current_data_root + 'ids.txt'

# Dataloader
dataset = coco224Dataset(data_path, labels_path, ids_path)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# Set up an experiment
experiment, exp_logger = config_experiment(current_experiment, resume=False, lr=learning_rate)
tfLogger = Logger('./tfLogs')

model = FeatureExtractor()
model.resnet.register_backward_hook(printgradnorm)
for block in model.children():
    if block.__class__.__name__=='UpsamplingBlock':
        for child in block.children():
            child.register_backward_hook(printgradnorm)
        continue
    block.register_backward_hook(printgradnorm)

if torch.cuda.is_available():
    model.cuda()

model.load_state_dict(experiment['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#optimizer.load_state_dict(experiment['optimizer_state_dict'])

current_epoch = experiment['epoch']
best_loss = experiment['best_loss']
loss_history = experiment['loss_history']

loss_fn = CostumeLoss()
loss_fn.register_backward_hook(printgradnorm)

exp_logger.info('training started/resumed at epoch ' + str(current_epoch))
for i in range(current_epoch, max_epoch_num):
    running_loss = 0

    for batch_num, batch in enumerate(dataloader):

        inputs = Variable(batch['image'].type(float_type))
        labels = batch['label'].numpy()
        features = model(inputs)
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1e06)
        #for p in model.state_dict():
            #print('===========\ngradient:{}\n----------{}\n'.format(p,model.state_dict()[p]))
            #break
        optimizer.zero_grad()
        current_loss = loss_fn(features,labels, k)
        np_loss = current_loss.data.numpy()
        if np.isnan(np_loss):
            print("nan loss---------------------------------------------")
            break
        current_loss.backward()
        optimizer.step()

        loss_history.append(np_loss)
        running_loss += np_loss
        tfLogger.scalar_summary('loss', np_loss[0], batch_num*(i+1))
        exp_logger.info('batch number '+str(batch_num)+', loss = '+str(np_loss[0]))

        if VISUALIZE:
            reduced_features = reduce(features.data)
            visualize(reduced_features)

        if batch_num%50 == 0:

            if best_loss is None or running_loss/50.0 < best_loss:
                best_loss = running_loss/50.0
                isBest = True
            else:
                isBest = False

            save_experiment({'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'epoch': i,
                             'best_loss': best_loss,
                             'loss_history': loss_history}, current_experiment, isBest)
            running_loss = 0







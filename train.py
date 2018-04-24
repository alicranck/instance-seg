from torch.autograd import Variable
import torch.autograd
from torchvision import datasets
from config import *
from costum_dataset import *
from torch.utils.data import DataLoader
from loss import CostumeLoss, sample_loss
from evaluate import *
#from logger import Logger
import logging

VISUALIZE = True

if torch.cuda.is_available():
    float_type = torch.cuda.FloatTensor
    double_type = torch.cuda.DoubleTensor
    int_type = torch.cuda.IntTensor
    long_type = torch.cuda.LongTensor

# Hyper parameters
k = 12
batch_size = 2
learning_rate = 0.001
lr_decay = 0.995
max_epoch_num = 100
current_experiment = 'test_03'
current_data_root = processed_train_root

# Paths to data, labels
data_path = voc_processed_images
labels_path = voc_processed_labels


def run():

    # Dataloader
    train_dataset = VOCDataset(data_path, labels_path, voc_train_ids)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = VOCDataset(data_path, labels_path, voc_val_ids)
    val_dataloader = DataLoader(val_dataset, batch_size)

    # Set up an experiment
    experiment, exp_logger = config_experiment(current_experiment, resume=True)
    #tfLogger = Logger('./tfLogs')

    model = FeatureExtractor()
    model.resnet.register_backward_hook(printgradnorm)
    for block in model.children():
        if block.__class__.__name__=='UpsamplingBlock':
            for child in block.children():
                child.register_backward_hook(printgradnorm)
            continue
        block.register_backward_hook(printgradnorm)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    model.load_state_dict(experiment['model_state_dict'])
    current_epoch = experiment['epoch']
    best_loss = experiment['best_loss']
    train_loss_history = experiment['train_loss']
    val_loss_history = experiment['val_loss']

    loss_fn = CostumeLoss()
    loss_fn.register_backward_hook(printgradnorm)

    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), learning_rate)

    exp_logger.info('training started/resumed at epoch ' + str(current_epoch))

    for i in range(current_epoch, max_epoch_num):
        adjust_learning_rate(optimizer, i, learning_rate, lr_decay)
        running_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            features = model(inputs)

            optimizer.zero_grad()
            current_loss = loss_fn(features,labels, k)
            current_loss.backward()
            optimizer.step()

            np_loss = current_loss.data.numpy()
            running_loss += np_loss
            exp_logger.info('epoch: '+ str(i) + ', batch number: '+str(batch_num)+', loss: '+str(np_loss[0]))

        train_loss = running_loss/batch_num

        # Evaluate model
        model.eval()
        val_loss, average_dice = evaluate_model(model, val_dataloader, loss_fn)

        if best_loss is None or val_loss < best_loss:
            best_loss = running_loss
            isBest = True
        else:
            isBest = False

        exp_logger.info('Saving checkpoint. Average validation loss is: ' + str(val_loss) +
                        'Average DICE is : ' + str(average_dice))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        save_experiment({'model_state_dict': model.state_dict(),
                         'epoch': i + 1,
                         'best_loss': best_loss,
                         'train_loss': train_loss_history,
                         'val_loss': val_loss_history}, current_experiment, isBest)

        plt.plot(train_loss_history, 'r')
        plt.plot(val_loss_history, 'b')
        os.makedirs('visualizations/' + current_experiment, exist_ok=True)
        plt.savefig('visualizations/' + current_experiment + '/loss.png')
        plt.close()

        if VISUALIZE:
            features = model(inputs)
            reduced_features = reduce(features.data, 10)
            visualize(inputs, reduced_features, current_experiment, i)

        model.train()


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


def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = np.power(lr, epoch)


if __name__=='__main__':
    run()

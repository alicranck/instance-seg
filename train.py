from torch.autograd import Variable
import torch.autograd
from config import *
from costum_dataset import *
from torch.utils.data import DataLoader
from loss import CostumeLoss, sample_loss
from evaluate import *


current_experiment = 'test_04'

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
    experiment, exp_logger = config_experiment(current_experiment, resume=True, context=context)

    fe = FeatureExtractor(context)
    classifier = ClassifyingModule(embedding_dim, classifier_hidden, num_classes)

    fe.resnet.register_backward_hook(printgradnorm)
    for block in fe.children():
        if block.__class__.__name__=='UpsamplingBlock':
            for child in block.children():
                child.register_backward_hook(printgradnorm)
            continue
        block.register_backward_hook(printgradnorm)

    classifier.register_backward_hook(printgradnorm)

    if torch.cuda.is_available():
        print("CUDA")
        fe.cuda()
        classifier.cuda()

    fe.load_state_dict(experiment['fe_state_dict'])
    classifier.load_state_dict(experiment['classifier_state_dict'])
    current_epoch = experiment['epoch']
    best_loss = experiment['best_loss']
    best_dice = experiment['best_dice']
    train_loss_history = experiment['train_loss']
    val_loss_history = experiment['val_loss']
    dice_history = experiment['dice']

    loss_fn = CostumeLoss()
    loss_fn.register_backward_hook(printgradnorm)

    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, fe.parameters()), learning_rate)

    exp_logger.info('training started/resumed at epoch ' + str(current_epoch))

    for i in range(current_epoch, max_epoch_num):
        adjust_learning_rate(optimizer, i, learning_rate, lr_decay)
        running_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            features = fe(inputs)

            optimizer.zero_grad()
            current_loss = loss_fn(features,labels, k)
            current_loss.backward()
            optimizer.step()

            np_loss = current_loss.data[0]
            running_loss += np_loss
            exp_logger.info('epoch: ' + str(i) + ', batch number: '+str(batch_num)+', loss: '+str(np_loss))

        train_loss = running_loss/batch_num

        # Evaluate model
        val_loss, average_dice = evaluate_model(fe, val_dataloader, loss_fn, current_experiment, i)

        if best_dice is None or average_dice < best_dice:
            best_dice = average_dice
            isBest = True
        else:
            isBest = False

        exp_logger.info('Saving checkpoint. Average validation loss is: ' + str(val_loss) +
                        ' Average DICE is : ' + str(average_dice))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        dice_history.append(average_dice)

        save_experiment({'fe_state_dict': fe.state_dict(),
                         'classifier_state_dict': classifier.state_dict(),
                         'epoch': i + 1,
                         'best_loss': best_loss,
                         'best_dice': best_dice,
                         'train_loss': train_loss_history,
                         'val_loss': val_loss_history,
                         'dice': dice_history}, current_experiment, isBest)

        plt.plot(train_loss_history, 'r')
        plt.plot(val_loss_history, 'b')
        os.makedirs('visualizations/' + current_experiment, exist_ok=True)
        plt.savefig('visualizations/' + current_experiment + '/loss.png')
        plt.close()
        plt.plot(dice_history)
        plt.savefig('visualizations/' + current_experiment + '/dice.png')
        plt.close()


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

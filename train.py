import torch.autograd
from costum_dataset import *
from torch.utils.data import DataLoader
from loss import CostumeLoss
from evaluate import *
from config import *


# Experiment name
current_experiment = 'test_05'

# Paths to data, labels
data_path = "/data/VOCdevkit/VOC2012/JPEGImages/"
labels_path = "/data/VOCdevkit/VOC2012/SegmentationObject/"
train_ids_path = "/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
val_ids_path = "/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"


def run():
    # Dataloader
    train_dataset = CostumeDataset(train_ids_path, data_path, labels_path, img_h=224, img_w=224)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = CostumeDataset(val_ids_path, data_path, labels_path, img_h=224, img_w=224)
    val_dataloader = DataLoader(val_dataset)

    # Set up an experiment
    experiment, exp_logger = config_experiment(current_experiment, resume=True, context=context)

    fe = FeatureExtractor(embedding_dim, context=context)

    fe.load_state_dict(experiment['fe_state_dict'])
    current_epoch = experiment['epoch']
    best_dice = experiment['best_dice']
    train_fe_loss_history = experiment['train_fe_loss']
    val_fe_loss_history = experiment['val_fe_loss']
    dice_history = experiment['dice_history']

    fe_loss_fn = CostumeLoss()
    fe_opt = torch.optim.Adam(filter(lambda p:p.requires_grad, fe.parameters()), learning_rate)

    exp_logger.info('training started/resumed at epoch ' + str(current_epoch))

    if torch.cuda.is_available():
        print("Using CUDA")
        fe.cuda()

    for i in range(current_epoch, max_epoch_num):
        adjust_learning_rate(fe_opt, i, learning_rate, lr_decay)
        running_fe_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            features = fe(inputs)
            fe_opt.zero_grad()
            fe_loss = fe_loss_fn(features, labels)
            fe_loss.backward()

            fe_opt.step()

            np_fe_loss = fe_loss.cpu().data[0]

            running_fe_loss += np_fe_loss
            exp_logger.info('epoch: ' + str(i) + ', batch number: ' + str(batch_num) +
                            ', loss: ' + "{0:.2f}".format(np_fe_loss))


        train_fe_loss = running_fe_loss/(batch_num+1)

        # Evaluate model
        fe.eval()
        val_fe_loss, avg_dice = evaluate_model(fe, val_dataloader, fe_loss_fn)
        fe.train()

        if best_dice is None or avg_dice > best_dice:
            best_dice = avg_dice
            isBest = True
        else:
            isBest = False

        exp_logger.info('Saving checkpoint. Average validation loss is: ' + "{0:.2f}".format(val_fe_loss) +
                        ', average DICE distance is: ' + "{0:.2f}".format(avg_dice))
        train_fe_loss_history.append(train_fe_loss)
        val_fe_loss_history.append(val_fe_loss)
        dice_history.append(avg_dice)

        # Save experiment
        save_experiment({'fe_state_dict': fe.state_dict(),
                         'epoch': i + 1,
                         'best_dice': best_dice,
                         'train_fe_loss': train_fe_loss_history,
                         'val_fe_loss': val_fe_loss_history,
                         'dice_history': dice_history}, current_experiment, isBest)

        # Plot and save loss history
        plt.plot(train_fe_loss_history, 'r')
        plt.plot(val_fe_loss_history, 'b')
        os.makedirs('visualizations/' + current_experiment, exist_ok=True)
        plt.savefig('visualizations/' + current_experiment + '/fe_loss.png')
        plt.close()

        # Plot and save loss history
        plt.plot(dice_history, 'r')
        os.makedirs('visualizations/' + current_experiment, exist_ok=True)
        plt.savefig('visualizations/' + current_experiment + '/dice.png')
        plt.close()

    return


def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr*np.power(decay_rate, epoch)


if __name__=='__main__':
    run()

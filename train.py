from torch.autograd import Variable
import torch.autograd
from config import *
from costum_dataset import *
from torch.utils.data import DataLoader
from loss import CostumeLoss
from evaluate import *
import torchvision


current_experiment = 'test_04'

# Paths to data, labels
data_path = voc_processed_images
labels_path = voc_processed_labels


def run():

    # Dataloader
    train_dataset = VOCDataset(data_path, labels_path, voc_train_ids)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = torchvision.datasets.CocoDetection(validation_images_path, val_annotations)
    val_dataloader = DataLoader(val_dataset)

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
    best_mAP = experiment['best_mAP']
    train_fe_loss_history = experiment['train_fe_loss']
    train_class_loss_history = experiment['train_class_loss']
    val_fe_loss_history = experiment['val_fe_loss']
    val_class_loss_history = experiment['val_class_loss']
    gtfg_mAP_history = experiment['gtfg_mAP_history']
    predfg_mAP_history = experiment['predfg_mAP_history']

    fe_loss_fn = CostumeLoss()
    classifier_loss_fn = nn.CrossEntropyLoss()

    fe_loss_fn.register_backward_hook(printgradnorm)
    classifier_loss_fn.register_backward_hook(printgradnorm)

    fe_opt = torch.optim.Adam(filter(lambda p:p.requires_grad, fe.parameters()), learning_rate)
    classifier_opt = torch.optim.Adam(filter(lambda p:p.requires_grad, classifier.parameters()), learning_rate)

    exp_logger.info('training started/resumed at epoch ' + str(current_epoch))

    for i in range(current_epoch, max_epoch_num):
        adjust_learning_rate(fe_opt, i, learning_rate, lr_decay)
        adjust_learning_rate(classifier_opt, i, learning_rate, lr_decay)

        running_fe_loss = 0
        running_class_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            class_labels = batch['class_label'].cpu().numpy()
            fg_masks = np.where(class_labels>0)
            features = fe(inputs)

            pred_masks = classifier(features)

            fe_opt.zero_grad()
            classifier_opt.zero_grad()
            fe_loss = fe_loss_fn(features, labels, k)
            class_loss = classifier_loss_fn(pred_masks, fg_masks)
            fe_loss.backward()
            class_loss.backward()
            fe_opt.step()
            classifier_opt.step()

            np_fe_loss = fe_loss.data[0]
            np_class_loss = class_loss.data[0]

            running_fe_loss += np_fe_loss
            running_class_loss += np_class_loss
            exp_logger.info('epoch: ' + str(i) + ', batch number: '+str(batch_num)+
                            ', fe loss: '+str(np_fe_loss) + ' class_loss: '+str(np_class_loss))


        train_fe_loss = running_fe_loss/batch_num
        train_class_loss = running_class_loss/batch_num

        # Evaluate model
        val_fe_loss, val_class_loss, gtfg_mAP, predfg_mAP = evaluate_model_coco(fe, classifier, val_dataloader,
                                                               fe_loss_fn, classifier_loss_fn, 300)

        if best_mAP is None or gtfg_mAP < best_mAP:
            best_mAP = gtfg_mAP
            isBest = True
        else:
            isBest = False

        exp_logger.info('Saving checkpoint. Average validation fe loss is: ' + str(val_fe_loss) + 'classifier loss: ' +
                       str(val_class_loss) + ' gtfg mAP is : ' + str(gtfg_mAP)+ ' predfg mAP is : ' + str(predfg_mAP))
        train_fe_loss_history.append(train_fe_loss)
        train_class_loss_history.append(train_class_loss)
        val_fe_loss_history.append(val_fe_loss)
        val_class_loss_history.append(val_class_loss)
        gtfg_mAP_history.append(gtfg_mAP)
        predfg_mAP_history.append(predfg_mAP)



        save_experiment({'fe_state_dict': fe.state_dict(),
                         'classifier_state_dict': classifier.state_dict(),
                         'epoch': i + 1,
                         'best_mAP': best_mAP,
                         'train_fe_loss': train_fe_loss_history,
                         'train_class_loss': train_class_loss_history,
                         'val_fe_loss': val_fe_loss_history,
                         'val_class_loss': val_class_loss_history,
                         'gtfg_mAP_history': gtfg_mAP_history,
                         'predfg_mAP_history': predfg_mAP_history}, current_experiment, isBest)

        plt.plot(train_fe_loss_history, 'r')
        plt.plot(val_fe_loss_history, 'b')
        os.makedirs('visualizations/' + current_experiment, exist_ok=True)
        plt.savefig('visualizations/' + current_experiment + '/fe_loss.png')
        plt.close()

        plt.plot(train_class_loss_history, 'r-')
        plt.plot(val_class_loss_history, 'b-')
        plt.savefig('visualizations/' + current_experiment + '/class_loss.png')
        plt.close()

        plt.plot(gtfg_mAP_history, 'g')
        plt.plot(predfg_mAP_history, 'y')
        plt.savefig('visualizations/' + current_experiment + '/mAP.png')
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
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr*np.power(decay_rate, epoch)


if __name__=='__main__':
    run()

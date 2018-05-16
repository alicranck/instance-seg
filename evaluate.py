import numpy as np
from torch.autograd import Variable
from scipy.misc import imsave
from torchvision.transforms import ToTensor
import hdbscan
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor, ToPILImage
from sklearn.manifold import TSNE
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
from skimage.transform import rescale
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.cocostuffhelper as helper
from scipy.ndimage import zoom
from config import *
import PIL.Image as im
from data_preprocessing import annsToSeg



def predict_label(features, downsample_factor=1):
    '''
    predicts a segmentation mask from the network output
    :param features: (c,h,w) ndarray containing the feature vectors outputted by the model
    :return: (h,w) ndarray with the predicted label (currently without class predictions
    '''
    features = np.transpose(features, [1,2,0])  # transpose to (h,w,c)
    features = skimage.measure.block_reduce(features, (downsample_factor,downsample_factor,1), np.max) #reduce resolution for performance

    h = features.shape[0]
    w = features.shape[1]
    c = features.shape[2]

    flat_features = np.reshape(features, [h*w,c])
    reduced_features = reduce(flat_features, 10)  # reduce dimension using PCA
    cluster_mask = cluster_features(reduced_features) # cluster with hDBSCAN
    #cluster_mask = determine_background(flat_features, cluster_mask)
    predicted_label = np.reshape(cluster_mask, [h,w])
    predicted_label = rescale(predicted_label, order=0, scale=downsample_factor, preserve_range=True)
    return np.asarray(predicted_label, np.int32)


def determine_background(features, cluster_mask):
    '''
    this function labels the cluster whose mean is closest to 0 as background
    :param features: a (n, d) ndarray of embeddings
    :param cluster_mask: a (n, 1) vector of cluster labels
    :return: an  (n, 1) vector of cluster labels, when the cluster with the smallest mean is labeled 0.
    '''
    min_mean = None
    background_label = 0
    cluster_mask += 1
    cluster_labels = np.unique(cluster_mask)

    for label in cluster_labels:
        if label==0:
            continue
        label_features = features[cluster_mask==label]
        cluster_mean_norm = np.linalg.norm(np.mean(label_features, 0), 2)

        if min_mean is None or cluster_mean_norm < min_norm:
            min_norm = cluster_mean_norm
            background_label = label

    cluster_mask[cluster_mask==background_label] = 0

    return cluster_mask


def cluster_features(features):
    '''
    this function takes a (h*w,c) numpy array, and clusters the c-dim points using MeanShift/DBSCAN.
    this function is meant to use for visualization and evaluation only.
    :param features: (h*w,c) array of h*w d-dim features extracted from the photo.
    :return: returns a (h*w,1) array with the cluster indices.
    '''
    # Define DBSCAN instance and cluster features
    dbscan = hdbscan.HDBSCAN(algorithm='boruvka_kdtree',min_cluster_size=500)
    labels = dbscan.fit_predict(features)
    #labels[np.where(labels==-1)] = 0

    return labels


def reduce(features, dimension=10):
    '''
    performs PCA dimensionality reduction on the input features
    :param features: a (n, d) or (h,w,d) numpy array containing the data to reduce
    :param dimension: the number of output channels
    :return: a (n, dimension) numpy array containing the reduced data.
    '''
    pca = PCA(n_components=dimension)
    pca_results = pca.fit_transform(features)
    return pca_results


def visualize(input, label, features, name, id):
    '''
    This function performs postprocessing (dimensionality reduction and clustering) for a given network
    output. it also visualizes the resulted segmentation along with the original image and the ground truth
    segmentation and saves all the images locally.
    :param input: (3, h, w) ndarray containing rgb data as outputted by the costume datasets
    :param label: (h, w) or (1, h, w) ndarray with the ground truth segmentation
    :param features: (c, h, w) ndarray with the embedded pixels outputted by the network
    :param name: str with the current experiment name
    :param id: an identifier for the current image (for file saving purposes)
    :return: None. all the visualizations are saved locally
    '''
    # Save original image
    os.makedirs('visualizations/' + name+'/segmentations', exist_ok=True)
    img_data = np.transpose(input, [1, 2, 0])
    max_val = np.amax(np.absolute(img_data))
    img_data = (img_data/max_val + 1) / 2  # normalize img
    imsave('visualizations/'+name+'/segmentations/' + str(id) + 'img.jpg', img_data)

    # Save ground truth
    if len(label.shape)==3:
        label = np.squeeze(label)
    label[np.where(label==255)] = 0
    label = label.astype(np.int32)
    imsave('visualizations/'+name+'/segmentations/' + str(id) + 'gt.png', label)

    # reduce features dimensionality and predict label
    predicted_label = predict_label(features, downsample_factor=2)
    imsave('visualizations/'+name+'/segmentations/' + str(id) + 'seg.png', predicted_label)

    # draw predicted seg on img and save
    plt.imshow(img_data)
    plt.imshow(predicted_label, alpha=0.5)
    plt.savefig('visualizations/'+name+'/segmentations/' + str(id) + 'vis.png')
    plt.close()

    return


def best_symmetric_dice(pred, gt):
    score1 = dice_score(pred, gt)
    score2 = dice_score(gt, pred)
    return max([score1, score2])


def dice_score(x, y):
    '''
    computes DICE of a predicted label and ground-truth segmentation. this is done for
    objects with no regard to classes.
    :param x: (1, h, w) or (h, w) ndarray
    :param y: (1, h, w) or (h, w) ndarrayruth segmentation segmentation
    :return: DICE score
    '''

    x_instances = np.unique(x)
    y_instances = np.unique(y)

    total_score = 0

    for x_instance in x_instances:
        max_val = 0
        for y_instance in y_instances:
            x_mask = np.where(x==x_instance, 1, 0)
            y_mask = np.where(y==y_instance, 1, 0)

            overlap = np.sum(np.logical_and(x_mask, y_mask))
            score = 2.0*overlap / np.sum(x_mask+y_mask)

            max_val = max([max_val, score])

        total_score += max_val

    return total_score/len(x_instances)


def evaluate_model(model, dataloader, loss_fn, name, epoch):
    '''
    evaluates average loss of a model on a given loss function, and average dice distance of
    some segmentations.
    :param model: the model to use for evaluation
    :param dataloader: a dataloader with the validation set
    :param loss_fn:
    :return: average loss, average dice distance
    '''
    running_loss = 0
    running_dice = 0
    for i, batch in enumerate(dataloader):
        dice_dist = 0
        inputs = Variable(batch['image'].type(float_type), volatile=True)
        labels = batch['label'].cpu().numpy()

        features = model(inputs)
        current_loss = loss_fn(features, labels)

        np_features = features.data.cpu().numpy()
        for j, item in enumerate(np_features):
            pred = predict_label(item, downsample_factor=2)
            dice_dist += best_symmetric_dice(pred, labels[j])

        running_loss += current_loss.data.cpu().numpy()[0]
        running_dice += dice_dist / (j+1)

    val_loss = running_loss / (i+1)
    average_dice = running_dice / (i+1)

    visualize(inputs.data[0].cpu().numpy(), labels[0], np_features[0], name, epoch)

    return val_loss, average_dice


def test_model(model, image_path, target_path, id):

    toTensor = ToTensor()
    img = im.open(image_path)

    old_size = img.size  # old_size is in (width, height) format
    new_size = (old_size[0] - (old_size[0]%32), old_size[1] - (old_size[1]%32))

    img = img.resize(new_size, im.ANTIALIAS)
    img_tensor = torch.unsqueeze(toTensor(img),0)

    features = model(img_tensor)
    np_features = features.data.cpu().numpy()

    predicted_label = predict_label(np_features, downsample_factor=1)
    imsave(target_path + str(id) + 'seg.png', predicted_label)

    # draw predicted seg on img and save
    plt.imshow(img)
    plt.imshow(predicted_label, alpha=0.5)
    plt.savefig(target_path + str(id) + 'vis.png')
    plt.close()


def evaluate_model_coco(fe_model, class_model, dataloader, fe_loss_fn, class_loss_fn, num_samples=400):

    cocoGt = COCO(val_annotations)
    toPil = ToPILImage()
    toTensor = ToTensor()
    img_ids = []
    coco_gtfg_detections = []
    coco_predfg_detections = []
    running_fe_loss = 0
    running_class_loss = 0

    for i, batch in enumerate(dataloader):
        img = toPil(batch[0][0])
        ann = batch[0][1]
        voc_label = annsToSeg(ann)
        img_ids.append(ann[0]['image_id'])

        old_size = img.size  # old_size is in (width, height) format
        new_size = (old_size[0] - (old_size[0] % 32), old_size[1] - (old_size[1] % 32))

        voc_label = zoom(voc_label, [new_size[1]/old_size[1], new_size[0]/old_size[0]], order=0)
        img = img.resize(new_size, im.ANTIALIAS)

        fg_label = np.where(voc_label>0, 1, 0)
        img_tensor = torch.unsqueeze(toTensor(img), 0)

        features = fe_model(Variable(img_tensor))
        fg_pred = class_model(features)

        fg_mask = np.where(fg_pred.data[0].numpy()[1]>fg_pred.data[0].numpy()[0], 1, 0)

        fe_loss = fe_loss_fn(features, np.expand_dims(voc_label,0))
        class_loss = class_loss_fn(fg_pred, Variable(torch.IntTensor(np.expand_dims(fg_label,0)).type(long_type)))

        running_fe_loss += fe_loss.data.cpu().numpy()
        running_class_loss += class_loss.data.cpu().numpy()

        np_features = features.data[0].cpu().numpy()

        pred = predict_label(np_features, downsample_factor=4)
        pred += 2

        gtfg_pred = pred * fg_label
        predfg_pred = pred * fg_mask
        gtfg_pred = zoom(gtfg_pred, [old_size[1]/new_size[1], old_size[0]/new_size[0]], order=0)
        predfg_pred = zoom(predfg_pred, [old_size[1]/new_size[1], old_size[0]/new_size[0]], order=0)

        plt.imshow(img)
        plt.imshow(gtfg_pred, alpha=0.4)
        plt.show()
        coco_gtfg_detections.extend(helper.segmentationToCocoResult(gtfg_pred, ann[0]['image_id'], 0))
        coco_predfg_detections.extend(helper.segmentationToCocoResult(predfg_pred, ann[0]['image_id'], 0))

        for d in coco_gtfg_detections:
            d['score'] = 0.8

        for d in coco_predfg_detections:
            d['score'] = 0.8

        if i==num_samples-1:
            break

    coco_gtfg_Dt = cocoGt.loadRes(coco_gtfg_detections)

    cocoEval = COCOeval(cocoGt, coco_gtfg_Dt, 'segm')
    cocoEval.params.imgIds = img_ids
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    gtfg_map = cocoEval.stats[0]

    coco_predfg_Dt = cocoGt.loadRes(coco_predfg_detections)

    cocoEval = COCOeval(cocoGt, coco_predfg_Dt, 'segm')
    cocoEval.params.imgIds = img_ids
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    predfg_map = cocoEval.stats[0]

    return running_fe_loss[0]/num_samples, running_class_loss[0]/num_samples, gtfg_map, predfg_map



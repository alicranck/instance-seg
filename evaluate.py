import numpy as np
import torch
from sklearn.cluster import MeanShift, DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
from skimage.transform import rescale
import os
from config import *


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
    cluster_mask = cluster_features(reduced_features)
    predicted_label = np.reshape(cluster_mask, [h,w])
    predicted_label = rescale(predicted_label, order=0, scale=downsample_factor, preserve_range=True)
    return np.asarray(predicted_label, np.int32)


def cluster_features(features):
    '''
    this function takes a (h*w,c) numpy array, and clusters the c-dim points using MeanShift/DBSCAN.
    this function is meant to use for visualization and evaluation only.
    :param features: (h*w,c) array of h*w d-dim features extracted from the photo.
    :return: returns a (h*w,1) array with the cluster indices.
    '''
    # Define MeanShift instance and cluster features
    #ms = MeanShift(bandwidth=0.5, cluster_all=False)
    #ms = DBSCAN(eps=0.15, min_samples=50)
    ms = hdbscan.HDBSCAN(algorithm='boruvka_kdtree',min_cluster_size=50)
    labels = ms.fit_predict(features)
    instances, counts = np.unique(labels, return_counts=True)

    # suppress small clusters
    for i, count in enumerate(counts):
        if count<100 or instances[i]==-1:
            labels = np.where(labels==instances[i], 0, labels)

    return labels


def reduce(features, dimension=10):
    '''
    performs PCA dimensionality reduction on the input features
    :param features: a (n, d) or (h,w,d) numpy array containing the data to reduce
    :param dimension: the number of output channels
    :return: a (n, dimension) numpy array containing the reduced data.
    '''
    #features = skimage.measure.block_reduce(features, (downsample,downsample,1), np.max) #reduce resolution for performance
    pca = PCA(n_components=dimension)
    pca_results = pca.fit_transform(features)
    print(np.sum(pca.explained_variance_ratio_))
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
    plt.imshow(img_data)  #convert to cpu on cloud
    plt.savefig('visualizations/' + name + '/segmentations/' + str(id) + 'img.png')
    plt.close()

    # Save ground truth
    if len(label.shape)==3:
        label = np.squeeze(label)
    plt.imshow(label)
    plt.savefig('visualizations/'+name+'/segmentations/' + str(id) + 'gt.png')
    plt.close()

    # reduce features dimensionality and predict label
    predicted_label = predict_label(features, downsample_factor=2)
    plt.imshow(predicted_label)
    plt.savefig('visualizations/'+name+'/segmentations/' + str(id) + 'seg.png')
    plt.close()
    return
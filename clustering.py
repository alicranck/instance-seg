from sklearn.cluster import MeanShift
import numpy as np



def cluster_features(features):
    '''
    this function takes a d*h*w numpy array, and clusters the d-dim points using MeanShift/DBSCAN.
    this function is meant to use for visualization and evaluation only.
    :param features: d*h*w array of h*w d-dim features extracted from the photo.
    :return: returns a h*w*1 array with the cluster indices.
    '''
    features = np.transpose(features, [1,2,0])
    h = features.shape[0]
    w = features.shape[1]
    d = features.shape[2]
    features = np.reshape(features, [h*w, d]) # Flatten features array

    # Define MeanShift instance and cluster features
    ms = MeanShift()
    labels = ms.fit_predict(features)
    print(labels) #debug

    labels = np.reshape(labels, [h*w*1])
    print(labels.shape) #debug

    return labels
from sklearn.cluster import DBSCAN
import numpy as np



def cluster_features(features):
    '''
    this function takes a d*h*w numpy array, and clusters the d-dim points using MeanShift.
    this function is meant to use for visualization and evaluation only.
    :param features: d*h*w array of h*w d-dim features extracted from the photo.
    :return: returns a h*w*1 array with the cluster indices.
    '''
    print (features.shape) #debug
    features = np.transpose(features, [1,2,0])
    print (features.shape) #debug
    h = features.shape[0]
    w = features.shape[1]
    d = features.shape[2]
    features = np.reshape(features, [h*w, d]) # Flatten features array

    # Define MeanShift instance and cluster features
    ms = DBSCAN(eps=0.3, min_samples=20)
    print(features.shape)
    labels = ms.fit_predict(features)
    print(labels) #debug

    labels = np.reshape(labels, [h*w*1])
    print(labels.shape) #debug

    return labels
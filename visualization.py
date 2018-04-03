from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import os
from config import *




def reduce(features):
    features = features.cpu()
    np_features = np.transpose(features[0].numpy(), [1,2,0])
    np_features = skimage.measure.block_reduce(np_features, (2,2,1), np.max)
    np_features = np.reshape(np_features, [112*112, embedding_dim])
    pca = PCA(n_components=15)
    pca_results = pca.fit_transform(np_features)
    print(np.sum(pca.explained_variance_ratio_))
    return pca_results


def visualize(inputs, reduced_features, name, current_batch):

    # Save original image
    os.makedirs('visualizations/' + name, exist_ok=True)
    img_data = np.transpose(inputs.cpu().data[0].numpy(), [1,2,0])
    max_val = np.amax(np.absolute(img_data))
    img_data = (img_data/max_val + 1) / 2  # normalize img
    plt.imshow(img_data)  #convert to cpu on cloud
    plt.savefig('visualizations/' + name + '/batch' + str(current_batch) + 'img.png')
    plt.close()

    # Embed predicted features to 3d and visualize
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(reduced_features)
    tsne_results = np.reshape(tsne_results,[112,112,3])
    tsne_results = (tsne_results/np.amax(np.absolute(tsne_results)) + 1) / 2
    plt.imshow(tsne_results)
    plt.savefig('visualizations/'+name+'/batch'+str(current_batch)+'.png')
    return








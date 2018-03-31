from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import skimage.measure
from config import *




def reduce(features):
    np_features = np.transpose(features[0].numpy(), [1,2,0])
    np_features = skimage.measure.block_reduce(np_features, (2,2,1), np.max)
    np_features = np.reshape(np_features, [112*112, embedding_dim])
    pca = PCA(n_components=5)
    pca_results = pca.fit_transform(np_features)
    print(np.sum(pca.explained_variance_ratio_))
    return pca_results


def visualize(reduced_features):
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(reduced_features)
    tsne_results = np.reshape(tsne_results,[112,112,3])
    #!!!!!!!!!!!normalize to [0,1] or [0,255]
    plt.imshow(tsne_results)
    plt.show()
    return







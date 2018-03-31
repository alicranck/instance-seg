import numpy as np
import torch
from torch.autograd import Variable
from config import *

eps = 0.0000000000001


class CostumeLoss(nn.Module):

    def __init__(self):
        super(CostumeLoss, self).__init__()

    def forward(self, features_batch, labels_batch, k):
        running_loss = Variable(torch.Tensor([0]).type(float_type))
        batch_size = features_batch.size()[0]
        for i in range(batch_size):
            running_loss += loss(features_batch[i], labels_batch[i], k)
            if np.isnan(running_loss.data.numpy()):
                print("------------------------------------------------------------------------------------")
                break
        return running_loss/batch_size


# loss goes to NaN after ~15 batches
def loss(features, label, k):

    sample, n_instances, instance_sizes = sample_vectors(features, label, k)
    if sample is None:
        return Variable(torch.Tensor([0]))
    mask, weights = get_masks_and_weights(k, n_instances, instance_sizes)
    distance_matrix = get_distances(sample)

    match_loss = torch.log(distance_matrix + eps) * mask
    mismatch_loss = torch.log(1+eps-distance_matrix) * (1-mask)

    pairwise_loss = weights * (match_loss + mismatch_loss)
    total_loss = (torch.sum(pairwise_loss) * (-1)) / len(sample)

    return total_loss.float()


# Needs more testing possibly
def sample_vectors(features, label, k):
    '''
    samples k feature vectors from the vectors corresponding to each instance
    :param features: a 3D tensor feature map sized (d, h, w)
    :param label: a label sized (h, w) where the [i,j] value is the id of the instance that
                    the (i,j) pixel belongs to
    :param k: number of vectors to sample form each instance
    :return: an (n, d) tensor containing k vectors per instance in the label, the number of instances,
                and their sizes
    '''
    sample = []
    unique, counts = np.unique(label, False, False, True)
    if len(unique)==1:
        return None, 0, []
    bad_indices = []   # This is a list of indices that should be excluded from the output

    for i in range(len(unique)):
        if unique[i] == 0:   # Ignore background
            bad_indices.append(i)
            continue

        locations = np.where(label == unique[i])
        sample_indices = np.random.choice(counts[i], k, replace=True)

        x = locations[0][sample_indices]
        y = locations[1][sample_indices]

        vectors = collect_vectors(features, x, y)
        sample.extend(vectors)

    unique = np.delete(unique, bad_indices)
    counts = np.delete(counts, bad_indices)
    return torch.stack(sample), len(unique), counts


# Tested on 3D float tensors, and autograd.Variables
def collect_vectors(tensor, x, y):
    '''
    gets a 3D input tensor of size (d, h, w) and collects the vectors in the locations
    specified in x, y. the collections is done from the (h, w) dimensions of the input
    :param tensor: an input tensor, which is treated as a matrix of (h,w) points in d-dimension space
    :param x: x coordinates of wanted vectors
    :param y: y coordinates of wanted vectors
    :return: returns a list of k tensors of dimension d where the i'th tensor in
                the list corresponds to input[:, x[i], y[i]]
    '''
    vectors = []
    for i in range(len(x)):
        vectors.append(tensor[:, x[i], y[i]])

    return vectors


# Tested on autograd.Variable
def get_distances(vectors):
    '''
    Gets as input a tensor(Variable) of size (n, d) and computes an (n,n) distance
    matrix D, where D[i,j] = sigma(input[i], input[j]). 'sigma' is a distance measure
    defined at https://arxiv.org/pdf/1703.10277.pdf
    :param vectors: a (n,d) tensor(Variable) representing n vectors of dimension d
    :return: a distance matrix of the input vectors.
    '''
    norms = torch.norm(vectors, 2, 1, keepdim=True)
    squared_norms = norms*norms

    distance_matrix = squared_norms - 2*torch.matmul(vectors, torch.t(vectors)) + torch.t(squared_norms)
    distance_matrix = torch.abs(distance_matrix)
    distance_matrix = distance_matrix.double()
    distance_matrix = 2.0 / (1 + torch.exp(distance_matrix))

    return distance_matrix


# Masks ok. weights seem good as well but might need another look
def get_masks_and_weights(k, num_instances, counts):

    n = k*num_instances
    At, A = np.mgrid[0:n, 0:n]

    ones = np.ones((n,n))
    zeroes = np.zeros((n,n))

    mask = np.zeros((n,n))
    weights = np.zeros((n,n))


    for i in range(num_instances):
        M_1 = np.where(((A>=k*i) & (A<k*(i+1))), ones, zeroes)
        M_2 = np.where(((At>=k*i) & (At<k*(i+1))), ones, zeroes)
        weights = weights + M_1*counts[i] + M_2*counts[i]
        mask = np.add(mask,  np.multiply(M_1,M_2))

    weights = 1.0/weights
    weights = weights/np.sum(weights)

    mask = Variable(torch.Tensor(mask).type(double_type))
    weights = Variable(torch.Tensor(weights).type(double_type))

    return mask, weights


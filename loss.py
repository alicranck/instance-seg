import numpy as np
import torch
from torch.autograd import Variable
from config import *


dv = 0
dd = 2.5

gamma = 0.005


class CostumeLoss(torch.nn.Module):

    def __init__(self):
        super(CostumeLoss, self).__init__()
        self.loss = contrasive_loss

    def forward(self, features_batch, labels_batch, k=0):
        running_loss = Variable(torch.Tensor([0]).type(float_type))
        batch_size = features_batch.size()[0]
        for i in range(batch_size):
            running_loss += self.loss(features_batch[i], labels_batch[i])
        return running_loss/batch_size


def contrasive_loss(features, label):

    label = label.flatten()
    features = features.permute(1,2,0).contiguous()
    shape = features.size()
    features = features.view(shape[0]*shape[1], shape[2])

    instances, counts = np.unique(label, False, False, True)
    if len(instances)==1: # no instances in image
        return Variable(torch.Tensor([0]).type(float_type))

    means = []
    var_loss = Variable(torch.Tensor([0]).type(double_type))
    dist_loss = Variable(torch.Tensor([0]).type(double_type))

    # calculate intra-cluster loss
    for instance in instances:
        if instance==255:   # ignore borders
            continue

        # collect all feature vector of a certain instance
        locations = Variable(torch.LongTensor(np.where(label == instance)[0]).type(long_type))
        vectors = torch.index_select(features,dim=0,index=locations).type(double_type)
        size = vectors.size()[0]

        # get instance mean and distances to mean of all points in an instance
        if instance == 0:   # push background to 0
            mean = Variable(torch.zeros((vectors.data.shape[1])).type(double_type))
            dist2mean = torch.sum(vectors**2, 1)
        else:
            mean = torch.sum(vectors, dim=0) / size
            dists = vectors - mean
            dist2mean = torch.sum(dists**2,1)

        var_loss += torch.sum(dist2mean)/size
        means.append(mean)

    # get inter-cluster loss - penalize close cluster centers
    means = torch.stack(means)
    num_clusters = means.data.shape[0]
    for i in range(num_clusters):
        if num_clusters==1:  # no inter cluster loss
            break
        for j in range(i+1, num_clusters):
            dist = torch.norm(means[i]-means[j])
            if dist.data[0]<dd*2:
                dist_loss += torch.pow(2*dd - dist,2)/(num_clusters-1)

    # regularization term
    reg_loss = torch.sum(torch.norm(means, 2, 1))

    total_loss = (var_loss + dist_loss + gamma*reg_loss) / num_clusters

    return total_loss.type(float_type)










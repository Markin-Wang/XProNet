import torch.nn as nn
class myTripletloss(nn.Module):
    def __init__(self,margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean'):

        super(myTripletloss, self).__init__()
        self.criterion = nn.TripletMarginLoss(margin, p, eps, swap, size_average, reduce, reduction)


    def forward(self,features):
        anchors = torch.mean(features,1)
        num_clusters = features.shape[0]
        positives = [[]*num_clusters]
        negatives = [[]*num_clusters]

        for i in range(num_clusters):
            dis_pow=anchors[i]
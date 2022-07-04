import torch
import torch.nn.functional as F

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


def my_con_loss(features, num_classes, num_protypes, labels, margin = 0.4, alpha = 1.5):
    B, _ = features.shape

    #labels = torch.arange(num_classes+2).expand(num_protypes, num_classes+2).t().flatten()
    #labels[(num_classes-1)*num_protypes:] = num_classes - 1

    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    losses = []
    for i in range(B):
        max_sim = labels[i].new_ones(B)
        pos_label_matrix = labels[:, labels[i] == 1]
        pos_label_matrix = torch.sum(pos_label_matrix, dim=1)
        pos_label_matrix[pos_label_matrix != 0] = 1
        label_diff = abs(labels[i] - labels[pos_label_matrix == 1]).sum(dim = 1)
        label_sum = (labels[i] + labels[pos_label_matrix == 1]).sum(dim=1)
        max_sim[pos_label_matrix == 1] = 1/(alpha ** (label_diff/label_sum))
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = max_sim - cos_matrix[i, :]
        pos_cos_matrix[pos_cos_matrix < 0] = 0
        neg_cos_matrix = cos_matrix[i, :] - margin
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        losses.append((pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum())
    loss = losses[0]
    for i in range(1, len(losses)):
        loss = loss + losses[i]
    loss /= (B * B)
    return loss

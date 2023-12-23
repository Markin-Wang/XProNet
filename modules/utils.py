import torch
import torch.nn.functional as F
import argparse
import torch.distributed as dist

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--exp_name', type=str, default='XPRONet',
                        help='the name of the experiments.')
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--use_amp', action='store_true', help='whether to enable mixed-precision training')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='data/iu_xray/labels.pickle',
                        help='the path to the directory containing the data.')

    parser.add_argument('--img_init_protypes_path', type=str, default='data/iu_xray/init_protypes_512.pt',
                        help='the path to the directory containing the data.')
    parser.add_argument('--init_protypes_path', type=str, default='data/iu_xray/init_protypes_512.pt',
                        help='the path to the directory containing the data.')

    parser.add_argument('--text_init_protypes_path', type=str, default='data/iu_xray/text_empty_initprotypes_512.pt',
                        help='the path to the directory containing the data.')
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr', 'cxr_gnome'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_txt_ebd', type=int, default=768, help='the dimension of extracted text embedding.')
    parser.add_argument('--d_img_ebd', type=int, default=512, help='the dimension of extracted img embedding.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank in DDP.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--trained_model_path', type=str, default='results/model_base.pth', help='the path to load the trained models.')
    parser.add_argument('--output', type=str, default='results', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--img_num_protype', type=int, default=10, help='.')
    parser.add_argument('--text_num_protype', type=int, default=10, help='.')
    parser.add_argument('--gbl_num_protype', type=int, default=10, help='.')
    parser.add_argument('--num_protype', type=int, default=10, help='.')
    parser.add_argument('--num_cluster', type=int, default=20, help='.')
    parser.add_argument('--start_eval_epoch', type=int, default=0, help='epoch to start validation')

    parser.add_argument('--weight_img_con_loss', type=float, default=1, help='.')
    parser.add_argument('--weight_txt_con_loss', type=float, default=1, help='.')

    parser.add_argument('--weight_img_bce_loss', type=float, default=1, help='.')
    parser.add_argument('--weight_txt_bce_loss', type=float, default=1, help='.')
    parser.add_argument('--img_con_margin', type=float, default=0.4, help='.')
    parser.add_argument('--txt_con_margin', type=float, default=0.4, help='.')
    parser.add_argument('--test_after', action='store_true', help='perform test after the training')

    args = parser.parse_args()
    return args

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

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
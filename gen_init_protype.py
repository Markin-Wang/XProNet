#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nnQ
import torchvision.models as models
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
import pickle
import numpy as np
from collections import Counter
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans


# In[ ]:


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='data/iu_xray/labels.pickle',
                        help='the path to the label annotatoin')


    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=32, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=8, help='the number of k.')
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
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
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
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--num_prototype', type=int, default=10, help='.')
    parser.add_argument('--num_cluster', type=int, default=20, help='.')
    parser.add_argument('--weight_cnn_loss', type=float, default=0.5, help='.')

    args = parser.parse_args()
    return args


# In[ ]:





# In[ ]:
num_classes = 13*3+1
num_cluster = 4
num_dim=512
initial_protypes = torch.zeros(((num_classes+2)*num_cluster,num_dim),dtype=float)
torch.nn.init.normal_(initial_protypes, 0, 1/num_dim)
counter=np.zeros(num_classes,dtype=float)

features_list = [[] for i in range(num_classes)]

# In[ ]:


args = parse_agrs()
tokenizer = Tokenizer(args)
model = models.resnet34(pretrained=True)
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)
train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True,drop_last=False)
model = model.cuda()
model.eval()
print('111',len(train_dataloader))

for images_id, images, reports_ids, reports_masks, labels in tqdm(train_dataloader):
    images = images.cuda()
    if args.dataset_name == 'iu_xray':                                                 
        features_1 = model(images[:,0])
        features_2 = model(images[:,1])
        features = (features_1+features_2)/2
    else:
        features = model(images)
    for i,image_id in enumerate(images_id):
        label = labels[i]
        counter[label == 1] += 1
        feature = features[i]
        feature = F.avg_pool2d(feature,kernel_size=7, stride=1, padding=0).squeeze()
        for j in range(num_classes):
            if label[j] == 1:
                features_list[j].append(feature.detach().cpu().numpy())
        #initial_protypes[label==1]+=feature.detach().cpu().numpy()



for i in range(num_classes):
    if len(features_list[i])==0:
        continue

    data = np.stack(features_list[i], 0)
    cluster_num = num_cluster if data.shape[0] > num_cluster else data.shape[0]
    if i == num_classes-1:
        cluster_num *= 3
    kmean_model = KMeans(n_clusters=cluster_num, max_iter=100,  init="k-means++")
    if len(features_list[i]) == 0:
        continue
    print(data.shape)
    results = kmean_model.fit_predict(data)
    label_pred = kmean_model.labels_
    for j in range(cluster_num):
        cluster_rep = np.mean(data[label_pred == j], 0)
        initial_protypes[i*num_cluster+j, :] = torch.from_numpy(cluster_rep)
    '''
    z = 0
    while cluster_num + z < num_cluster:
        initial_protypes[i * num_cluster + cluster_num + z, :] = initial_protypes[i * num_cluster + z, :]
        z += 1
        '''



print((~torch.isfinite(initial_protypes)).sum())
'''

for i in range(len(initial_protypes)):
    if counter[i]==0:
        continue
    initial_protypes[i]=initial_protypes[i]/counter[i]
print(sum(counter==0))
with open('./init_prototypes_512.pickle','wb') as myfile:
    pickle.dump(initial_protypes,myfile) 
'''

torch.save(initial_protypes, "./init_protypes_2048_224_noflip.pt")
        
        
        
        
    


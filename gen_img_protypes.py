import torch
import torch.nn as nn
import torchvision.models as models
from modules.tokenizers import Tokenizer
import numpy as np
from collections import Counter
from modules.utils import parse_agrs
import json
from modules.dataloaders import R2DataLoader
from tqdm import tqdm
import os

import torch.distributed as dist
import torch.nn.functional as F
from sklearn.cluster import KMeans


def main():
    args = parse_agrs()
    world_size = args.n_gpu

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    num_classes = 14
    num_cluster = 10
    num_dim = 2048
    initial_protypes = torch.zeros((num_classes * num_cluster, num_dim), dtype=float)
    torch.nn.init.normal_(initial_protypes, 0, 1 / num_dim)
    counter = np.zeros(num_classes, dtype=float)

    features_list = [[] for i in range(num_classes)]

    # In[ ]:

    if args.dataset_name == 'cxr_gnome':
        tokenizer = None
    else:
        tokenizer = Tokenizer(args)

    model = models.resnet101(pretrained=True)
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, drop_last=False)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False,
                                                      find_unused_parameters=False)
    model.eval()
    print('111', len(train_dataloader))

    visual_features = {}

    # uncomment this when generate the visual feature dict
    for images_id, images, reports_ids, reports_masks, labels in tqdm(train_dataloader):
        images = images.cuda()
        if args.dataset_name == 'iu_xray':
            features_1 = model(images[:, 0])
            features_2 = model(images[:, 1])
            features = (features_1 + features_2) / 2
        else:
            features = model(images)
        for i, image_id in enumerate(images_id):
            feature = features[i]
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.detach().cpu().numpy()
            visual_features[image_id] = feature

    torch.save(visual_features, './visual_features_mimic.pth')
    exit()


    for images_id, images, reports_ids, reports_masks, labels in tqdm(train_dataloader):
        images = images.cuda()
        if args.dataset_name == 'iu_xray':
            features_1 = model(images[:, 0])
            features_2 = model(images[:, 1])
            features = (features_1 + features_2) / 2
        else:
            features = model(images)
        for i, image_id in enumerate(images_id):
            label = labels[i]
            counter[label == 1] += 1
            feature = features[i]
            feature = F.avg_pool2d(feature, kernel_size=7, stride=1, padding=0).squeeze()
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.detach().cpu().numpy()
            for j in range(num_classes):
                if label[j] == 1:
                    features_list[j].append(feature)
            # initial_protypes[label==1]+=feature.detach().cpu().numpy()

    for i in range(num_classes):
        if len(features_list[i]) == 0:
            continue

        data = np.stack(features_list[i], 0)
        cluster_num = num_cluster if data.shape[0] > num_cluster else data.shape[0]
        # if i == num_classes - 1:
        #     cluster_num *= 3
        kmean_model = KMeans(n_clusters=cluster_num, max_iter=100, init="k-means++")
        if len(features_list[i]) == 0:
            continue
        print(data.shape)
        results = kmean_model.fit_predict(data)
        label_pred = kmean_model.labels_
        for j in range(cluster_num):
            cluster_rep = np.mean(data[label_pred == j], 0)
            initial_protypes[i * num_cluster + j, :] = torch.from_numpy(cluster_rep)
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


if __name__ == '__main__':
    main()

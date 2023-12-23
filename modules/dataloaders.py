import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
from .dataset_arrow import CXRGenomeDatasetArrow


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle,drop_last=False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'cxr_gnome':
            self.dataset = CXRGenomeDatasetArrow(args=self.args, tokenizer=self.tokenizer, split=self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        self.sampler = torch.utils.data.DistributedSampler(
            self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=self.shuffle
        )

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'sampler': self.sampler,
            #'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last':drop_last,
            'prefetch_factor': self.batch_size // self.num_workers * 2
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch, label = zip(*data)
        image_batch = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)
        label_batch = torch.stack(label, 0)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        return image_id_batch, image_batch, torch.LongTensor(target_batch), \
               torch.FloatTensor(target_masks_batch), label_batch
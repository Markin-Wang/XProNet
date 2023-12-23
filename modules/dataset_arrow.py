import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

import pandas as pd

import numpy as np
import pyarrow as pa
import io
from modules.tokenizers import Tokenizer
import torchvision


class BaseDatasetArrow(Dataset):
    def __init__(self, args, tokenizer, split, transform=None, text_column_name='caption', name=None):
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = args.max_seq_length
        self.dataset_name = args.dataset_name
        root = args.image_dir
        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{root}/{name}.arrow", "r")).read_all()
        self.text_column_name = text_column_name
        # self.all_texts = self.table[text_column_name].to_pandas().tolist()
        self.all_texts = self.table[text_column_name].to_pandas()

        if split == 'train':
            self.tokenizer = Tokenizer(args, self.all_texts)

        self.labels_path = args.label_path
        self.labels = json.loads(open(self.labels_path, 'r').read())[split]


    def get_raw_image(self, index, image_key="image"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        if 'iu_xray' in self.dataset_name:
            image1 = self.get_raw_image(index, image_key='image1')
            image2 = self.get_raw_image(index, image_key='image2')
            image_tensor1 = [self.transform(image1)]
            image_tensor2 = [self.transform(image2)]
            image_tensor = torch.stack(image_tensor1 + image_tensor2,dim=0)
        else:
            image = self.get_raw_image(index, image_key=image_key)
            image_tensor = self.transform(image)
        # image = self.get_raw_image(index, image_key=image_key)
        # image_tensor = [tr(image) for tr in self.transforms]
        iid = self.table['image_id'][index].as_py()
        if 'iu_xray' in self.dataset_name:
            array = iid.split('-')
            modified_id = array[0] + '-' + array[1]
            label = np.array(self.labels[modified_id]).astype(np.float32)
        else:
            label = torch.FloatTensor(self.labels[iid])

        return {
            "image": image_tensor,
            "img_id": iid,
            "img_index": index,
            "raw_index": index,
            'img_labels': label,
        }

    def get_text(self, index):
        text = self.all_texts[index][0] # only one gt caption in rrg
        encoding = self.tokenizer(text)[:self.max_seq_length]
        mask = [1] * len(encoding)
        gt_text = self.all_texts[index]
        seq_length = len(encoding)
        return {
            "text": encoding,
            "img_index": index,
            "cap_index": index,
            "mask": mask,
            "raw_index": index,
            "gt_txt": gt_text,
            "seq_length": seq_length,
        }

    def get_suite(self, index):
        ret = dict()
        ret.update(self.get_image(index))
        txt = self.get_text(index)
        ret.update(txt)
        return ret

    def __len__(self):
        return len(self.all_texts)


class IuxrayMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "iu_xray_train"
        elif split == "val":
            name = "iu_xray_val"
        elif split == "test":
            name = "iu_xray_test"
        else:
            name = None

        super().__init__(*args, **kwargs, split=split,name=name, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split or 'val' in self.split:
            iid = self.table["image_id"][index].as_py()
            # iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        sample = (suite['img_id'], suite['image'], suite['text'], suite['mask'], suite['seq_length'], suite['label'])

        return sample


class CXRGenomeDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "cxr_gnome_train_ft_atsg"
        elif split == "val":
            name = "cxr_gnome_val_ft_sg"
        elif split == "test":
            name = "cxr_gnome_test_ft_sg"
        else:
            name = None

        super().__init__(*args, **kwargs, split=split, name=name, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split or 'val' in self.split:
            iid = self.table["image_id"][index].as_py()
            # iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        sample = (suite['img_id'], suite['image'], suite['text'], suite['mask'], suite['seq_length'], suite['img_labels'])
        return sample

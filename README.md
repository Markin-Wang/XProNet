# R2GenCMN

This is the implementation of [Cross-modal Memory Networks for Radiology Report Generation]() at ACL-IJCNLP-2021.

## Citations

If you use or extend our work, please cite our paper at ACL-IJCNLP-2021.
```
@inproceedings{chen-acl-2021-r2gencmn,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and
      Shen, Yaling  and
      Song, Yan and
      Wan, Xiang",
    booktitle = "Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing",
    month = aug,
    year = "2021",
}
```

## Requirements

- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`


## Download R2GenCMN
You can download the models we trained for each dataset from [here](https://github.com/cuhksz-nlp/R2GenCMN/blob/main/data/r2gencmn.md).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

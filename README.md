# XProNet

This is the official implementation of [Cross-modal Prototype Driven Network for Radiology Report Generation]() accepted to ECCV2022.

## Abstract

Radiology report generation (RRG) aims to describe auto-
matically a radiology image with human-like language. As an alternative
to expert diagnosis, RRG could potentially support the work of radiol-
ogists, reducing the burden of manual reporting. Previous approaches
often adopt an encoder-decoder architecture and focus on single-modal
feature learning, while few studies explore cross-modal feature inter-
action. Here we propose a Cross-modal PROtotype driven NETwork
(XPRONET) to promote cross-modal pattern learning and exploit it
to improve the task of radiology report generation. This is achieved by
three well-designed, fully differentiable and complementary modules: a
shared cross-modal prototype matrix to record the cross-modal proto-
types; a cross-modal prototype network to learn the cross-modal pro-
totypes and embed the cross-modal information into the visual and
textual features; and an improved multi-label contrastive loss to en-
able and enhance multi-label prototype learning. Experimental results
demonstrate that XPRONET obtains substantial improvements on two
commonly used medical report generation benchmark datasets, i.e., IU-
Xray and MIMIC-CXR, where its performance exceeds recent state-of-
the-art approaches by a large margin on IU-Xray dataset and achieves
the SOTA performance on MIMIC-CXR. 

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

## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.6]
- [PyTorch = 1.6]
- [Torchvision]
- [Pycocoevalcaption]

* You can create the environment via conda:
```bash
conda env create --name [env_name] --file env.yml
```


## Download Trained Models
You can download the trained models [here](https://drive.google.com/drive/folders/1_y_6srL2ZnvDvE_I0YDvdgRzZCNrcMUf?usp=sharing).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

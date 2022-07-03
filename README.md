# XProNet

This is the official implementation of [Cross-modal Memory Networks for Radiology Report Generation]() accepted to ECCV2022.

## Abstract

The core for tackling the fine-grained visual categorization (FGVC) is to learn subtleyet discriminative features. Most previous works achieve this by explicitly selecting thediscriminative parts or integrating the attention mechanism via CNN-based approaches.However,  these  methods  enhance  the  computational  complexity  and  make  the  modeldominated  by  the  regions  containing  the  most  of  the  objects.   Recently,  vision  trans-former (ViT) has achieved SOTA performance on general image recognition tasks.  Theself-attention mechanism aggregates and weights the information from all patches to theclassification token,  making it perfectly suitable for FGVC. Nonetheless,  the classifi-cation  token  in  the  deep  layer  pays  more  attention  to  the  global  information,  lackingthe local and low-level features that are essential for FGVC. In this work, we proposea novel pure transformer-based framework Feature Fusion Vision Transformer (FFVT)where we aggregate the important tokens from each transformer layer to compensate thelocal, low-level and middle-level information.  We design a novel token selection mod-ule called mutual attention weight selection (MAWS) to guide the network effectivelyand efficiently towards selecting discriminative tokens without introducing extra param-eters.  We verify the effectiveness of FFVT on four benchmarks where FFVT achievesthe state-of-the-art performance.

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

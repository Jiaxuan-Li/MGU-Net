
# MGU-Net
Multi-Scale GCN-Assisted Two-Stage Network for Joint Segmentation of Retinal Layers and Disc in Peripapillary OCT Images

The codes are implemented in PyTorch and trained on NVIDIA Tesla V100 GPUs.

## Introduction
An accurate and automated tissue segmentation algorithm for retinal optical coherence tomography (OCT) images is crucial for the diagnosis of glaucoma. However, due to the presence of the optic disc, the anatomical structure of the peripapillary region of the retina is complicated and is challenging for segmentation. To address this issue, we develop a novel graph convolutional network (GCN)-assisted two-stage framework to simultaneously label the nine retinal layers and the optic disc. Specifically, a multi-scale global reasoning module is inserted between the encoder and decoder of a U-shape neural network to exploit anatomical prior knowledge and perform spatial reasoning. We conduct experiments on human peripapillary retinal OCT images. We also provide public access to the [collected dataset](http://www.yuyeling.com/project/mgu-net/), which might contribute to the research in the field of biomedical image processing. The Dice score of the proposed segmentation network is 0.820 ± 0.001 and the pixel accuracy is 0.830 ± 0.002, both of which outperform those from other state-of-the-art techniques.
<div align=center><img width="750" src="https://github.com/Jiaxuan-Li/MGU-Net/blob/main/figs/fig2.png"/></div>

## Experiments
### Dataset
1. Collected dataset: Download our [collected dataset](http://www.yuyeling.com/project/mgu-net/).
2. Public dataset: Download [Duke SD-OCT dataset](http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)

### Train and test 
Run the following script to train and test the two-stage model.
```
python main_ts.py --name tsmgunet -d ./data/dataset --batch-size 1 --epoch 50 --lr 0.001
```

## Results
### Results on the collected dataset
<div align=center><img width="700" src="https://github.com/Jiaxuan-Li/MGU-Net/blob/main/figs/fig3.png"/></div>

### Results on the public dataset
<div align=center><img width="700" src="https://github.com/Jiaxuan-Li/MGU-Net/blob/main/figs/fig4.png"/></div>

For more details, please refer to our [paper](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-12-4-2204).

## Citation
If you use the codes or collected dataset for your research, please cite the following paper:
```
@article{Li:21,
author = {Jiaxuan Li and Peiyao Jin and Jianfeng Zhu and Haidong Zou and Xun Xu and Min Tang and Minwen Zhou and Yu Gan and Jiangnan He and Yuye Ling and Yikai Su},
journal = {Biomed. Opt. Express},
number = {4},
pages = {2204--2220},
title = {Multi-scale GCN-assisted two-stage network for joint segmentation of retinal layers and discs in peripapillary OCT images},
volume = {12},
year = {2021},
url = {http://www.osapublishing.org/boe/abstract.cfm?URI=boe-12-4-2204},
doi = {10.1364/BOE.417212},
}
```

## Acknowledgements
The codes are built on [AI-Challenger-Retinal-Edema-Segmentation](https://github.com/ShawnBIT/AI-Challenger-Retinal-Edema-Segmentation) and [GloRe](https://github.com/facebookresearch/GloRe). We sincerely appreciate the authors for sharing their codes.

## Contact
If you have any questions, please contact jiaxuan.li@sjtu.edu.cn

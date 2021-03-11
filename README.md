
# MGU-Net

Multi-Scale GCN-Assisted Two-Stage Network for Joint Segmentation of Retinal Layers and Disc in Peripapillary OCT Images

The code is implemented in PyTorch and trained on NVIDIA Tesla V100 GPUs.

## Introduction
An accurate and automated tissue segmentation algorithm for retinal optical coherence tomography (OCT) images is crucial for the diagnosis of glaucoma. However, due to the presence of the optic disc, the anatomical structure of the peripapillary region of the retina is complicated and is challenging for segmentation. To address this issue, we developed a novel graph convolutional network (GCN)-assisted two-stage framework to simultaneously label the nine retinal layers and the optic disc. Specifically, a multi-scale global reasoning module is inserted between the encoder and decoder of a U-shape neural network to exploit anatomical prior knowledge and perform spatial reasoning. We conducted experiments on human peripapillary retinal OCT images. The Dice score of the proposed segmentation network is 0.820 ± 0.001 and the pixel accuracy is 0.830 ± 0.002, both of which outperform those from other state-of-the-art techniques.

<div align=center><img width="750" src="https://github.com/Jiaxuan-Li/MGU-Net/blob/main/figs/fig2.png"/></div>

## Experiments
### Dataset
1. Collected dataset: Download our collected dataset in this link (it will be available soon!!!).
2. Public dataset: [Duke SD-OCT dataset](http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)

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

## Citation
If you use this code or dataset for your research, please cite the following papers:
```
@misc{li2021multiscale,
  title={Multi-scale GCN-assisted two-stage network for joint segmentation of retinal layers and disc in peripapillary OCT images}, 
  author={Jiaxuan Li and Peiyao Jin and Jianfeng Zhu and Haidong Zou and Xun Xu and Min Tang and Minwen Zhou and Yu Gan and Jiangnan He and Yuye Ling and Yikai Su},
  journal={arXiv preprint arXiv:2102.04799},
  year={2021},
}

@inproceedings{10.1117/12.2582905,
  author = {Jiaxuan Li and Yuye Ling and Jiangnan He and Peiyao Jin and Jianfeng Zhu and Haidong Zou and Xun Xu and Shuo Shao and Yu Gan and Yikai Su},
  title = {{A GCN-assisted deep learning method for peripapillary retinal layer segmentation in OCT images}},
  booktitle = {Optical Coherence Tomography and Coherence Domain Optical Methods in Biomedicine XXV},
  year = {2021}
}
```

## Acknowledgements
This code is built on [AI-Challenger-Retinal-Edema-Segmentation](https://github.com/ShawnBIT/AI-Challenger-Retinal-Edema-Segmentation) and [GloRe](https://github.com/facebookresearch/GloRe). We thank the authors for sharing their codes.

## Contact
If you have any questions, please contact jiaxuan.li@sjtu.edu.cn

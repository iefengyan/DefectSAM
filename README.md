# DefectSAM: Hierarchically Adapting SAM for Pixel-Wise Surface Defect Detection



This official repository contains the source code of DefectSAM. (accepted by TNNLS 2025).

Segment anything model (SAM) has recently demonstrated powerful segmentation ability for natural scene images (NSIs). However, the SAM exhibits limited performance in defect detection owing to the weak appearance of defects and cluttered backgrounds in industrial images. In this article, we propose a hierarchically adapting SAM for pixel-wise surface defect detection, named DefectSAM, which effectively modulates and decodes multilevel features of the encoder to capture defect information. Specifically, we introduce a learnable feature adaptation component between the image encoder and the decoder to modulate each level of features via the dual-feature adaptation unit. The dual-feature adaptation unit mainly includes the correlation-gated feature adaptation (CGFA) module and the mask-guided feature adaptation (MGFA) module. The CGFA exploits cross correlation spatial gating maps to adaptively incorporate a convolutional feature pyramid and Transformer features during feature adaptation, which is beneficial for capturing defect details. Moreover, the MGFA utilizes the mask prediction of high-level features as semantic guidance to select top-confidence foreground and background tokens for feature adaptation, focusing more on defect details and suppressing background noise. Extensive experiments on three defect detection datasets (i.e., MVTec AD, CrackSeg9k, ZJU-Leaper, and Magnetic tile) demonstrate that the proposed method achieves state-of-the-art performance with few learnable parameters, which greatly improves the generalization of SAM in defect detection.

<img width="1029" height="572" alt="image" src="https://github.com/user-attachments/assets/cd6ac975-c9bb-4a45-b6ee-d39aa4a2d5f9" />


Our code is based on mmsegmentation.

<!-- [[Project Page]](https://denseclip.ivg-research.xyz/)  -->
[[paper]](https://xplorestaging.ieee.org/document/10844993)

## Usage

### Requirements

- torch>=1.8.0
- torchvision
- timm
- mmcv-full==1.3.17
- mmsegmentation==0.19.0
- mmdet==2.17.0
- regex
- ftfy
- fvcore
- tqdm==4.62.3
- pysodmetrics==1.3.0
- imageio==2.9.0

To use our code, please first install the `mmcv-full` and `mmseg` following the official guidelines ([`mmseg`](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md)) and prepare the datasets accordingly. 

### Pre-trained Uni-Perceiver Models

Download the pre-trained Uni-Perceiver models ([repo](https://github.com/fundamentalvision/Uni-Perceiver)                                              | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf)) and save it to the `pretrained` folder.


### Training & Evaluation

To train the FGSA-Net model, run:

```
bash dist_train.sh configs/COD/fgsa_net_512.py 1
```

To evaluate the performance of FGSA-Net, run:

```
bash dist_test.sh configs/COD/fgsa_net_512.py /path/to/checkpoint 1
```



## Citation
If you find our work useful in your research, please consider citing:
```
@ARTICLE{10844993,
  author={Zhang, Shizhou and Kong, Dexuan and Xing, Yinghui and Lu, Yue and Ran, Lingyan and Liang, Guoqiang and Wang, Hexu and Zhang, Yanning},
  journal={IEEE Transactions on Multimedia}, 
  title={Frequency-Guided Spatial Adaptation for Camouflaged Object Detection}, 
  year={2025},
  volume={27},
  pages={72-83},
  doi={10.1109/TMM.2024.3521681}}
```

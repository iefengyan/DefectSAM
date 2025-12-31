# DefectSAM: Hierarchically Adapting SAM for Pixel-Wise Surface Defect Detection

This official repository contains the source code of DefectSAM. (accepted by TNNLS2025).

Segment anything model (SAM) has recently demonstrated powerful segmentation ability for natural scene images (NSIs). However, the SAM exhibits limited performance in defect detection owing to the weak appearance of defects and cluttered backgrounds in industrial images. In this article, we propose a hierarchically adapting SAM for pixel-wise surface defect detection, named DefectSAM, which effectively modulates and decodes multilevel features of the encoder to capture defect information. Specifically, we introduce a learnable feature adaptation component between the image encoder and the decoder to modulate each level of features via the dual-feature adaptation unit. The dual-feature adaptation unit mainly includes the correlation-gated feature adaptation (CGFA) module and the mask-guided feature adaptation (MGFA) module. The CGFA exploits cross correlation spatial gating maps to adaptively incorporate a convolutional feature pyramid and Transformer features during feature adaptation, which is beneficial for capturing defect details. Moreover, the MGFA utilizes the mask prediction of high-level features as semantic guidance to select top-confidence foreground and background tokens for feature adaptation, focusing more on defect details and suppressing background noise. Extensive experiments on three defect detection datasets (i.e., MVTec AD, CrackSeg9k, ZJU-Leaper, and Magnetic tile) demonstrate that the proposed method achieves state-of-the-art performance with few learnable parameters, which greatly improves the generalization of SAM in defect detection.

<img width="1257" height="678" alt="image" src="https://github.com/user-attachments/assets/042b5b49-7a50-42a7-8a62-92f3fe656637" />


## Citation
If you find our work useful in your research, please consider citing:
```
@article{yan2025defectsam,
  author={Yan, Feng and Jiang, Xiaoheng and Lu, Yang and Cao, Jiale and Xu, Mingliang},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={DefectSAM: Hierarchically Adapting SAM for Pixel-Wise Surface Defect Detection}, 
  year={2025},
  volume={36},
  number={10},
  pages={18830-18843}}
```
# Acknowledgement
We would like to acknowledge the contributions of public projects, such as [SAM-Adaper](https://github.com/tianrun-chen/SAM-Adapter-PyTorch), whose code has been utilized in this repository.

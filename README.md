# Window Setting Optimization
This is the reference implementation of the window setting optimization (WSO) layer for medical image deep learning.

Hyunkwang Lee, Myeongchan Kim, and Synho Do. Practical Window Setting Optimiza-tion for Medical Image Deep Learning arXiv preprint arXiv:1812.00572. 2018 Dec 3. <br/>
arXiv link: https://arxiv.org/abs/1812.00572

## Abstract
The recent advancements in deep learning have allowed for numerous applications in computed tomography (CT), with potential to improve diagnostic accuracy, speed of interpretation, and clinical efficiency. However, the deep learning community has to date neglected window display settings - a key feature of clinical CT interpretation and opportunity for additional optimization. Here we propose a window setting optimization (WSO) module that is fully trainable with convolutional neural networks (CNNs) to find optimal window settings for clinical performance. Our approach was inspired by the method commonly used by practicing radiologists to interpret CT images by adjusting window settings to increase the visualization of certain pathologies. Our approach provides optimal window ranges to enhance the conspicuity of abnormalities, and was used to enable performance enhancement for intracranial hemorrhage and urinary stone detection. On each task, the WSO model outperformed models trained over the full range of Hounsfield unit values in CT images, as well as images windowed with pre-defined settings. The WSO module can be readily applied to any analysis of CT images, and can be further generalized to tasks on other medical imaging modalities.

## How to use
- Download images and models
- Define WSO layer with your own model (see HowToUse.ipynb or WindowsOpt.py)
- Deploy trained model in the original paper (see Deploy_code_WindowsOptimizer.ipynb or deploy.py)
  
## Requirement for test codes
- Keras=2.2.4  
- tensorflow-gpu=1.12.0  
- openCV=3.4.2   


## Citing windows optimization
Please cite windows optimization in your publications if it helps your research:

```
@article{2018arXiv181200572L,
  title={Practical Window Setting Optimization for Medical Image Deep Learning},
  author={{Lee}, Hyunkwang and {Kim}, Myeongchan and {Do}, Synho},
  journal={arXiv preprint arXiv:1812.00572},
  year={2018},
  month = {Dec},
}
```

# A<sup>3</sup> (Augument Any Annotations): Inpaint Anything Meets Object Detection Annotation

<b>Nonpublished Technology</b>

TL; DR: This is a customised version of Inpaint Anything. Users can set annotation file and augument annotations by segmentation of SAM and inpainting of SD.

Original Repos as below.

<p>
<a href="https://github.com/geekyutao/Inpaint-Anything">
<img src="https://img.shields.io/badge/-InpaintingAnything-181717.svg?logo=github&style=flat">
 </a>
<a href="https://github.com/facebookresearch/segment-anything/">
<img src="https://img.shields.io/badge/-SegmentAnythingModel-181717.svg?logo=github&style=flat">
</a>
<a href="https://github.com/CompVis/stable-diffusion/">
<img src="https://img.shields.io/badge/-StableDiffusion-181717.svg?logo=github&style=flat">
</p>
 
 ## Installation

 ```bash
 git clone 
 cd 
 ```


 Details are on each original repos.

 ```bash
 conda create -n a3 python=3.9
 conda activate a3
 conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
 pip install git+https://github.com/facebookresearch/segment-anything.git
 python -m pip install diffusers transformers accelerate scipy safetensors
 ```

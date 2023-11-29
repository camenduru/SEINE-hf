# SEINE
This repository is the official implementation of [SEINE](https://arxiv.org/abs/2310.20700).

**[SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction](https://arxiv.org/abs/2310.20700)**

[Arxiv Report](https://arxiv.org/abs/2310.20700) | [Project Page](https://vchitect.github.io/SEINE-project/)

<img src="seine.gif" width="800">


##  Setups for Inference

### Prepare Environment
```
conda env create -f env.yaml
conda activate seine
```

### Downlaod our model and T2I base model
Download our model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1cWfeDzKJhpb0m6HA5DoMOH0_ItuUY95b?usp=sharing) and save to directory of ```pre-trained```


Our model is based on Stable diffusion v1.4, you may download [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) to the director of ``` pre-trained ```

Now under `./pretrained`, you should be able to see the following:
```
├── pretrained_models
│   ├── seine.pt
│   ├── stable-diffusion-v1-4
│   │   ├── ...
└── └── ├── ...
        ├── ...
```

#### Inference for I2V 
```python
python sample_scripts/with_mask_sample.py --config configs/sample_i2v.yaml
```
The generated video will be saved in ```./results/i2v```.

#### Inference for Transition
```python
python sample_scripts/with_mask_sample.py --config configs/sample_transition.yaml
```
The generated video will be saved in ```./results/transition```.



#### More Details
You can modify ```./configs/sample_mask.yaml``` to change the generation conditions.
For example, 
```ckpt``` is used to specify a model checkpoint.
```text_prompt``` is used to describe the content of the video.
```input_path``` is used to specify the path to the image.


## BibTeX
```bibtex
@article{chen2023seine,
title={SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction},
author={Chen, Xinyuan and Wang, Yaohui and Zhang, Lingjun and Zhuang, Shaobin and Ma, Xin and Yu, Jiashuo and Wang, Yali and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
journal={arXiv preprint arXiv:2310.20700},
year={2023}
}
```
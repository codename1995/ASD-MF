# SALIENCY PREDICTION VIA MULTI-LEVEL FEATURES AND DEEP SUPERVISION FOR CHILDREN WITH AUTISM SPECTRUM DISORDER

This repository contains Keras implementation of our atypical visual saliency prediction model.

## Cite
Please cite with the following Bibtex code:
```
@inproceedings{wei2019saliency,
  title={Saliency prediction via multi-level features and deep supervision for children with autism spectrum disorder},
  author={Wei, Weijie and Liu, Zhi and Huang, Lijin and Nebout, Alexis and Le Meur, Olivier},
  booktitle={2019 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={621--624},
  year={2019},
  organization={IEEE}
}
```

## Pretrained weight on [Saliency4ASD](https://saliency4asd.ls2n.fr/)
[Google Drive](https://drive.google.com/file/d/1bK3CYLf_SVAmg1BMhgZgJ6fSDmQSgnkz/view?usp=sharing)

## Training
Train model from scratch
```bash
$ python train.py --train_set_path path/to/training/set --val_set_path path/to/validation/set 
```
For training model based on our pretrained weight, please download the weight file and put it into `weights/`.
```bash
$ python train.py --train_set_path path/to/training/set --val_set_path path/to/validation/set --model_path weights/weights--1.4651.pkl
```
The dataset directory structure should be 
```
└── Set  
    ├── Images  
    │   ├── 1.png  
    │   └── ...
    ├── FixMaps  
    │   ├── 1.png  
    │   └── ...
    ├── maps  
        ├── 1.mat  
        └── ...
```

## Testing
Clone this repository and download the pretrained weights.

Then just run the code using 
```bash
$ python test.py --images_path path/to/test/images --results_path path/to/results --model_path path/to/saved/models
```
This will generate saliency maps for all images in the images directory and save them in results directory

## Requirements:
cuda 8.0  
cudnn 5.1  
python	3.5  
keras	2.2.2  
theano	0.9.0  
opencv	3.1.0  
matplotlib	2.0.2  

## Acknowledgement
The code is heavily inspired by the following project:
1. SAM : https://github.com/marcellacornia/sam

Thanks for their contributions.

## Contact 
If any question, please contact codename1995@shu.edu.cn

## License 
This code is distributed under MIT LICENSE.

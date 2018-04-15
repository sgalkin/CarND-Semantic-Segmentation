# Semantic Segmentation

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/55a50d32bef74eea9128116cbf84b863)](https://www.codacy.com/app/tech.svg/CarND-T3P2?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sgalkin/CarND-T3P2&amp;utm_campaign=Badge_Grade)

### Overview
In this project, implements Fully Convolutional Network (FCN) for semantic
segmentation of a road in an image. Architecture on the FCN described in [1].

### Goal
The project labels at least _80%_ of the road and label no more than _20%_
of non-road pixels as road. The model doesn't have to predict correctly all the
images, just most of them.

### Demo
Sample image
![](https://github.com/sgalkin/CarND-T3P2/blob/master/docs/um_000032.png)

The full collection of testing dataset could be found [here](https://mega.nz/#F!Fo9lWBYB!50QXprRFfldAm2Rahr_ptg)

### Usage
```sh
python main.py
```

### Project structure
* `main.py` - the project entry point
* `helper.py` - helper functions (download model, inference, etc)
* `project_tests.py` - basic tests of correctness

### Dependencies
#### Language
* [`Python 3`](https://www.python.org/) - Python is a programming language that lets you work quickly
and integrate systems more effectively

#### Tools
* [`Conda`](https://conda.io) - Package, dependency and environment management for any language—Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN

#### Libraries and Frameworks
* [`TensorFlow`](https://www.tensorflow.org/) - An open source machine learning framework for everyone
* [`NumPy`](http://www.numpy.org/) - The fundamental package for scientific computing with Python
* [`scikit-image`](http://scikit-image.org) - Image processing in Python
* [`tqdm`](https://github.com/tqdm/tqdm) - A fast, extensible progress bar for Python and CLI

#### Dataset
* [`Kitti Road dataset`](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip)

### Run
0. Clone [the](https://github.com/sgalkin/CarND-T3P2.git) repository.
1. Download dataset.
2. Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.
3. Setup environment `conda env create -f environment.yml carnd-t3p2`
4. Run `python main.py`

### Implementation
#### Architecture
The project uses `FCN-8` Neural Network Architecture as described in [1].
* Encoder - `VGG-16` [2]
* Decoder
  * restores original image resolution
  * adds skip connection for _3rd_ layer
  * adds skip connection for _4th_ layer
  * _L2_ regularization used with weight _1e-3_ for each transpose convolution
  * All weights initialized using _truncated normal distribution_ with standard
    deviation _1e-2_

#### Data Preparaion
* Dataset was augmented by flipping all images from left to right

#### Training
* Batch size - _34_
* Number of epochs - _75_
* Initial learining rate - _0.0005_
* Optimizer - ADAM
* Encoder weights _frozen_

#### Rational
* _L2_ regularization helps to prevent overfitting since the train dataset is
  very small
* _Frozen_ encoder weight - prevent overfitting since the train dataset is
  very small, increases speed of training
* _Data augmentation_ - helps improve accuracy and prevent overfitting

#### Results
* Loss - _0.114276_
  ![](https://github.com/sgalkin/CarND-T3P2/blob/master/docs/Loss.png)
* IoU - _0.904379_
  ![](https://github.com/sgalkin/CarND-T3P2/blob/master/docs/IoU.png)
* Weights - available [here](https://mega.nz/#F!godnVTTb!A183NNrVFcXho2qkbl2zmg)

## Reference
1. J. Long, E. Shelhamer, T. Darrell, "Fully convolutional networks for semantic segmentation", 2014. [arXiv:1605.06211](https://arxiv.org/pdf/1605.06211.pdf)
2. Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman. [arXiv:1409.1556](https://arxiv.org/pdf/1409.1556.pdf)

# Assignment_10_Resnet_Utils
Code to help with Data loading, Augmentation, Training of CIFAR10 Dataset

## Folder Structure

~~~
    Assignment_10_Resnet_Utils
    |──config
    |── ── assignment_10.yaml
    |── data_loader
    |── ── data_augmentation.py
    |── ── data_loader.py
    |── utils
    |── ── helper.py
    |──── learning_rate_finder.py
    |── ── visulatization.py
    |── model
    |── ── custom_resnet.py
    |── ── train_test.py
    |── README.md

~~~

## Config File
Includes the configurations for augmentations, training and learning rates.

### [data_loader](https://github.com/prarthanats/Assignment_10_Resnet_Utils/tree/main/dataload)

Data Loader function downloads,calculate dataset statistics and transform the data and performs data augmentation using albumentations to create test and train loaders. The various augmentations we have done are  [Augmentation](https://github.com/prarthanats/Assignment_10_Resnet_Utils/blob/main/dataload/data_augmentation.py)
1. Normalize: Normalizes the image by subtracting the mean and dividing by the standard deviation.
2. PadIfNeeded: Pads the image if its height or width is smaller than the specified minimum height or width.
3. RandomCrop: Randomly crops the image to the specified height and width.
4. HorizontalFlip: Flips the image horizontally.
5. Cutout: Applies random cutout augmentation by removing rectangular regions of the image.
6. ToTensorV2: Converts the image to a PyTorch tensor.

## Utils

#### [Helper Class](https://github.com/prarthanats/Assignment_10_Resnet_Utils/blob/main/utils/helper.py)
It contains some of the miscellaneous functions to:
1. Check for presence of GPU in runtime and create a device accordingly
2. Loading configuration variables from yaml file as a dictionary
3. Reading the model summary

#### [Learning Rate Finder](https://github.com/prarthanats/Assignment_10_Resnet_Utils/blob/main/utils/learning_rate_finder.py)
Learning Rate Finder uses the torch_lr_finder library to identify the maximum learning rate for the training data

#### [Visulatization](https://github.com/prarthanats/Assignment_10_Resnet_Utils/blob/main/utils/visulatization.py)
It contains some of visulation functions for:
1. Plotting misclassified images with labels alongside images
2. Plotting loss and accuracy metrics post training
3. Plotting Class Specific images

## Model

#### [Custom Resnet](https://github.com/prarthanats/Assignment_10_Resnet_Utils/blob/main/model/custom_resnet.py)

The Custom Resnet model consists of several components:

PrepBlock: The PrepBlock is responsible for processing the initial input data. In the given code, it consists of a series of operations applied to the input image. It applies a 3x3 convolutional layer with 3 input channels (RGB image) and 64 output channels, followed by a ReLU activation function, batch normalization, and dropout.

ConvolutionBlock: The ConvBlock represents a convolutional block that consists of one or more convolutional layers followed by pooling, batch normalization, and activation functions. In the given code, the ConvBlock includes a 3x3 convolutional layer with a specified number of input and output channels, followed by a 2x2 max pooling layer, batch normalization, and ReLU activation function. This block is used to extract features from the input data.

ResidualBlock: The ResidualBlock implements a residual block, which is commonly used in deep neural networks to enable better gradient flow during training. It consists of two 3x3 convolutional layers with the same number of input and output channels, followed by batch normalization and ReLU activation functions. The output of the block is obtained by adding the input to the result of the second convolutional layer, which creates a residual connection. This allows the network to learn residual mappings, helping with the optimization process and improving the network's performance.

#### [Train and Test](https://github.com/prarthanats/Assignment_10_Resnet_Utils/blob/main/model/train_test.py)

Training and testing codes to input optimizers, schedulers and loss criteria.

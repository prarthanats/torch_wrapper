# Torch Wrapper Utils for EVA Course

Code to help with Data loading, Augmentation, Training and Visulaization

## Folder Structure

~~~
    Assignment_10_Resnet_Utils
    |──config
    |── ── assignment_10.yaml
    |── ── assignment_11.yaml
    |── model
    |── ── custom_resnet.py
    |── ── resnet.py
    |── utils
    |── ── data_augmentation.py
    |── ── data_handeling.py
    |── ── data_loader.py
    |── ── gradcam.py
    |── ── helper.py
    |── ── train_test.py
    |── ── visulaization.py
    |── main.py
    |── README.md

~~~

## [Config File](https://github.com/prarthanats/torch_wrapper/blob/main/config)
Includes the configurations for various assignments, which follows the following structure
~~~
    1. Model and Model parameters
    2. Data Augmentation
    3. Data Loader Configurations
    4. Criterion, Optimizer
    5. Learning Rate Scheduler
    6. Training Parameters
~~~

## [Models for Assignments](https://github.com/prarthanats/torch_wrapper/tree/main/model)

Includes the model files for various assignments

## Utils
Includes the utility files for various assignments such as 

#### [Augmentation](https://github.com/prarthanats/torch_wrapper/blob/main/utils/data_augmentation.py)

The data augmentation library used for CIFAR data is albumentations package 

#### [Data Handling](https://github.com/prarthanats/torch_wrapper/blob/main/utils/data_handeling.py)

The Handling of data for dataset, dataloaders and dataset_statistics

#### [Data Loader](https://github.com/prarthanats/torch_wrapper/blob/main/utils/data_loader.py)

Data Loader function downloads, created data arguments using the configuration from config file

#### [Grad Cam](https://github.com/prarthanats/torch_wrapper/blob/main/utils/gradcam.py)

Gradcam output for explaining the model output. Referenced the code from [Grad Cam Library](https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py)

#### [Helper](https://github.com/prarthanats/torch_wrapper/blob/main/utils/helper.py)

Helper function includes dunctionality for set seed functionality and process_config functionality to process the configuration file

#### [Train and Test](https://github.com/prarthanats/torch_wrapper/blob/main/utils/train_test.py)

Training and testing codes to input optimizers, schedulers and loss criteria.

#### [Visulatization](https://github.com/prarthanats/torch_wrapper/blob/main/utils/visulaization.py)
It contains some of visulation functions for:
1. Plotting misclassified images with labels alongside images
2. Plotting loss and accuracy metrics post training
3. Plotting Class Specific images


## [Main File](https://github.com/prarthanats/torch_wrapper/blob/main/main.py)

Main file includes the TriggerTraining Class that includes call
~~~
    1. Data loader
    2. Set the device
    3. Model Summary
    4. Learning Rate finder
    5. Run Training
    6. Get wrong predictions
    7. Plot the misclassified
~~~

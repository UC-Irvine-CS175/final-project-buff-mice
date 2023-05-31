[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/lzU1yAcG)
# Getting Started: Downloading Data from the Cloud & PyTorch Custom Datasets, Transforms, and DataLoader
## Tony Yin


[![Dataset, Augmentations, and Data Utils Tests](https://github.com/UC-Irvine-CS175/download-dataset-augmentations-dataloader-Boracle-02/actions/workflows/hw_dataset.yml/badge.svg)](https://github.com/UC-Irvine-CS175/download-dataset-augmentations-dataloader-Boracle-02/actions/workflows/hw_dataset.yml)

## Assignment Overview
In this homework we will:
- [ ] Setup your machine with the tools necessary using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) environments.
- [ ] Write data utility functions using [`boto3`](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html) Amazon Web Services (AWS) SDK for Python that allows you to interact with data stored on the cloud.
- [ ] Partition the available training data into a train and validation set by subsetting `meta.csv`.
- [ ] Write a custom PyTorch [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) class using the [NASA BPS Microscopy Data](https://registry.opendata.aws/bps_microscopy/).
- [ ] Write custom PyTorch [Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset) to augment the dataset.
- [ ] Write the retrieve packaged image and label data using the [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader), a PyTorch iterator which allows for batching, shuffling, and loading the data in parallel using multiprocessing workers.
## Setting Up Your Project Environment
To make sure you download all the requirements to begin this homework assignment go to `setup/environments` and follow the directions to setup your conda environment based on whether your machine is gpu or cpu enabled using Miniconda, a lightweight form of Anaconda.

## Retrieving Public Dataset Using AWS SDK

The BPS dataset is a public data repository hosted in an S3 AWS bucket which is a cloud-based storage service provided by Amazon. It allows users to store and retrieve data from anywhere at any time. S3 stands for Simple Storage Service, and it provides a scalable and reliable way to store data in the cloud. AWS S3 buckets can store any type of object or file, such as documents, images, videos, and other types of data, and can be accessed through APIs, web interfaces, or command-line interfaces. In this assignment you will be interacting with this API using `boto3`, an AWS package that allows you to use Python code to interact with data stored in the S3 bucket.

### AWS CLI Tool
Using the AWS command line interface (note it has to be installed) you can quickly assess what lives inside of the S3 bucket.

`aws s3 ls --no-sign-request s3://nasa-bps-training-data/Microscopy/`

results in:

`PRE train/`

To look inside the `train/` directory, you may type:

`aws s3 ls --no-sign-request s3://nasa-bps-training-data/Microscopy/train --recursive`

which gives you a full listing of the files. To copy all files locally you may run:

`aws s3 cp --no-sign-request s3://nasa-bps-training-data/Microscopy/train --recursive <your_local_folder_here>`

**Since we love learning new things and want to add AWS API knowledge, we will use `boto3` instead.**

### AWS boto3 API
`boto3` is a python library that allows you to interact with AWS services using python code. To get started, we need to initialize the boto3 client with the s3 service and the signature_version set to UNSIGNED to allow for public access to the data:

`s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))`

To download the data from cloud to our local machine, we use the `download_file` method from the boto3 client. Alternatively, we can use the `get_object` method from the boto3 client to get the data as a stream without having to save files locally. In either case, in order to retrieve data, the bucket name and s3 path to the data are required. In this assignment we will do both. 

1. We will retrieve `meta.csv` as a stream in order to select a subset of the data based on the attributes `dose_Gy` and `hr_post_exposure` and save this into a new metadata csv file.
2. Using the new metadata csv file, we will selectively download image files of interest from the s3 bucket.

## Writing Custom Datasets, DataLoaders, and Transforms in PyTorch
This part of the assignment is adapted from PyTorch's tutorial linked below:
[Writing Custom Datasets, DataLoaders, and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

### The Dataset Class
To define a custom dataset, we need to inherit the Dataset class from PyTorch. We do this by creating a new class and overriding the following methods:
- `__init__`
- `__len__`
- `__getitem__`

The `__init__` method is run once when instantiating the Dataset object and requires information such as the location of the data, transforms to apply, etc. The `__len__` method returns the number of samples in our dataset. The `__getitem__` method loads and returns a sample from the dataset at the given index idx. Based on the index, it identifies the image’s location on disk, fetches and reads the image, applies the transforms, and returns the tensor image following transformation and corresponding label in a tuple.

### Transformations
We will augment the data by writing transformatioons as callable classes instead of functions that we can later use with the PyTorch Compose class to string together sequences of transformations on the images. The transforms you will write are the following:
- NormalizeBPS: Normalizes uint16 to float32 between 0-1
- ResizeBPS: Resizes images
- VFlipBPS: Vertically flips the image
- HFlipBPS: Horizontally flips the image
- RotateBPS: Rotates the image [90, 180, 270]
- RandomCropBPS: Randomly crops the image
- ToTensor: Converts a np.array image to a PyTorch Tensor (final transformation)

Using the `torchvision.transforms.Compose(tranforms:list[Tranform])` class we can specify the list of tranforms/augmentations that an image can undergo. 

Augmentations are important for deep learning because they enhance the robustness of the model since it recieves variations on examples.
### DataLoader
The `torch.utils.data.DataLoader` is an iterator that allows you to batch the data (take more than one image and label at a time), shuffle the data, and load the data in parallel using multiprocessing. An example of how to call the dataloader is below:  

`dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)`

### Some Notes on PyTorch Tensors
Tensors are similar to numpy's ndarrays, with the exception that they add an additional dimension to images. For example, you may have a numpy array with the dimensions (height, width, channels), where channels correspond to RGB. The Tensor representation of the image will have an additional dimension of (batch_size, channels, height, width). This is why the `ToTransform` transformation will be the last to do a final conversion of the image from numpy arry to Tensor prior to calling the DataLoader.
## Files to Work On
- `src/dataset/bps_dataset.py`
- `src/dataset/augmentation.py`
- `src/data_utils.py`

## NOTE
- It is required that you add your name and github actions workflow badge to your readme.
- Check the logs from github actions to verify the correctness of your program.
- The initial code will not work. You will have to write the necessary code and fill in the gaps.
- Commit all changes as you develop the code in your individual private repo. Please provide descriptive commit messages and push from local to your repository. If you do not stage, commit, and push git classroom will not receive your code at all.
- Make sure your last push is before the deadline. Your last push will be considered as your final submission.
- There is no partial credit for code that does not run.
- If you need to be considered for partial grade for any reason (failing tests on github actions,etc). Then message the staff on discord before the deadline. Late requests may not be considered.

## References
[GH Badges](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge)

"""
This file is based on the following tutorial: 
https://learnopencv.com/t-sne-for-feature-visualization/
"""
import os
import random
import numpy as np
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))

import pandas as pd
import cv2
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torchvision.models import resnet
from torchvision.models.resnet import Bottleneck
from torch.hub import load_state_dict_from_url
from mpl_toolkits.mplot3d import Axes3D


from src.dataset.bps_datamodule import BPSDataModule
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)


from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50
from torchvision.models import squeezenet1_1

from src.data_utils import save_tiffs_local_from_s3
import boto3
from botocore import UNSIGNED
from botocore.config import Config


@dataclass
class BPSConfig:
    """ Configuration options for BPS Microscopy dataset.

    Args:
        data_dir: Path to the directory containing the image dataset. Defaults
            to the `data/processed` directory from the project root.

        train_meta_fname: Name of the training CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_train.csv'

        val_meta_fname: Name of the validation CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_test.csv'
        
        save_dir: Path to the directory where the model will be saved. Defaults
            to the `models/SAP_model` directory from the project root.

        batch_size: Number of images per batch. Defaults to 4.

        max_epochs: Maximum number of epochs to train the model. Defaults to 3.

        accelerator: Type of accelerator to use for training.
            Can be 'cpu', 'gpu', 'tpu', 'ipu', 'auto', or None. Defaults to 'auto'
            Pytorch Lightning will automatically select the best accelerator if
            'auto' is selected.

        acc_devices: Number of devices to use for training. Defaults to 1.
        
        device: Type of device used for training, checks for 'cuda', otherwise defaults to 'cpu'

        num_workers: Number of cpu cores dedicated to processing the data in the dataloader

        dm_stage: Set the partition of data depending to either 'train', 'val', or 'test'
                    However, our test images are not yet available.


    """
    data_dir:           str = os.path.join(root, 'data', 'processed')
    train_meta_fname:   str = 'meta_train.csv'
    val_meta_fname:     str = 'meta_test.csv'
    save_vis_dir:       str = os.path.join(root, 'models','dummy_vis')
    save_models_dir:    str = os.path.join(root, 'models','baselines')
    batch_size:         int = 64
    max_epochs:         int = 3
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 4
    dm_stage:           str = 'train'
    

#num classes = 
class RadiationClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(RadiationClassifier, self).__init__()
        self.squeeze_net = squeezenet1_1(pretrained=True)
        self.squeeze_net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.squeeze_net.features(x)
        x = self.squeeze_net.classifier(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        # num_features = my_model.fc.in_features
        # x = nn.Linear(num_features, 5)
        return x

def main():
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_meta_fname = "meta.csv"


    data_dir = root / 'data'
    # testing get file functions from s3
    local_train_dir = data_dir / 'processed'

    config = BPSConfig()
    # Instantiate BPSDataModule
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   resize_dims=(64, 64),
                                   meta_csv_file = s3_meta_fname,
                                    meta_root_dir=s3_path,
                                    s3_client= s3_client,
                                    bucket_name=bucket_name,
                                    s3_path=s3_path,
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    bps_datamodule.prepare_data()

   
if __name__ == '__main__':
    main()
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
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from torchvision.models import squeezenet1_1


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
    data_dir:           str = root / 'data' / 'processed'
    train_meta_fname:   str = 'meta_train.csv'
    val_meta_fname:     str = 'meta_test.csv'
    save_vis_dir:       str = root / 'models' / 'dummy_vis'
    save_models_dir:    str = root / 'models' / 'baselines'
    batch_size:         int = 64
    max_epochs:         int = 3
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 4
    dm_stage:           str = 'train'
    


class RadiationClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(RadiationClassifier, self).__init__()
        self.resnet = resnet.resnet34(weights=ResNet34_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

def main():
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = BPSConfig()
    # Instantiate BPSDataModule
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   resize_dims=(64, 64),
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage=config.dm_stage)

    # Creating the model
    model = RadiationClassifier()

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # added to fix overfitting 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    # Epochs 
    num_epochs = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate through the batches of images from the dataloader
        for batch_idx, (images, labels) in tqdm(enumerate(bps_datamodule.train_dataloader()), desc="Running model inference"):
            images = images.to(device) 
            labels = np.argmax(labels, axis=1)
            labels = labels.to(device)

            images = images.repeat(1, 3, 1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_loss = running_loss / len(bps_datamodule.train_dataset)
        train_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Updating learning rate 
        scheduler.step(train_loss)

        if epoch == 5:
            model_path = "all5_resnet34_CNN_model_1.pth"
            torch.save(model.state_dict(), model_path)
            print("Trained model saved at:", model_path)

        # if epoch == 1:
        #     # Save the trained model
        #     model_path = "alldata_CNN_model_epoch1.pth"
        #     torch.save(model.state_dict(), model_path)
        #     print("Trained model saved at:", model_path)

        # if epoch == 5:
        #     # Save the trained model
        #     model_path = "alldata_CNN_model_epoch5.pth"
        #     torch.save(model.state_dict(), model_path)
        #     print("Trained model saved at:", model_path)

        # if epoch == 10:
        #     # Save the trained model
        #     model_path = "alldata_CNN_model_epoch10.pth"
        #     torch.save(model.state_dict(), model_path)
        #     print("Trained model saved at:", model_path)

        # if epoch == 25:
        #     # Save the trained model
        #     model_path = "alldata_CNN_model_epoch25.pth"
        #     torch.save(model.state_dict(), model_path)
        #     print("Trained model saved at:", model_path)
    
    # Save the trained model
    model_path = "all6_resnet34_CNN_model_1.pth"
    torch.save(model.state_dict(), model_path)
    print("Trained model saved at:", model_path)

    
    # bps_datamodule.setup(stage="validate")
    # device = torch.device("cpu")
    
    # with torch.no_grad():
    #     model.eval()
    #     test_loss = 0.0
    #     test_correct = 0
    #     test_total = 0

    #     for batch_idx, (images, labels) in tqdm(enumerate(bps_datamodule.val_dataloader()), desc="Running model test"):

    #         images= images.to(device)
    #         labels = np.argmax(labels, axis=1)
    #         labels = labels.to(device)

    #         # Repeat the single channel to have 3 channels
    #         images = images.repeat(1, 3, 1, 1)

    #         outputs = model(images)
    #         loss = criterion(outputs, labels)

    #         _, predicted = torch.max(outputs.data, 1)

    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    #         test_loss += loss.item()

    #     test_loss /= 100
    #     test_accuracy = test_correct / test_total

    #     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
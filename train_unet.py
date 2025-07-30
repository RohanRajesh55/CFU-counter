import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json

# --- Configuration ---
DATASET_DIR = 'dataset'
TRAINING_LISTS_DIR = 'dataset/training_lists'
MODEL_SAVE_PATH = 'unet_model.pth'
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 4
IMAGE_SIZE = (128, 128)
# ---------------------



class ConvBlock(nn.Module):
    """(Convolution => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


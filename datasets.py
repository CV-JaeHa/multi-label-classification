""" 폴더 이름
1. dirty_mnist.zip라는 zip파일을 dirty_mnist라는 이름으로 압축 풀기
2. test_dirty_mnist.zip라는 zip파일을 test_dirty_mnist라는 폴더에 압축 풀기
"""

# Import Library
import pandas as pd
import numpy as np
import cv2
import os
import PIL
import random
import glob
import time
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)

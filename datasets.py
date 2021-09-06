""" 폴더 이름
1. dirty_mnist.zip라는 zip파일을 dirty_mnist라는 이름으로 압축 풀기
2. test_dirty_mnist.zip라는 zip파일을 test_dirty_mnist라는 폴더에 압축 풀기
"""

# Import Library
"""
import glob
import torch. cv2
import numpy as np
import os, random, tqdm, pickle
import pandas as pd
"""
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
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASGSEED"] = str(seed)
print(f"seed : {seed}")

# Set Path
""" 해당 경로는 다음과 같은 구조를 가정합니다.
data
└── dirty_mnist
     ├── model
     └── data
          ├── train_data
          │   └── 00000~49999.png
          └── test_data
               └── 50000~54999.png
"""

os.chdir("./data/dirty_mnist")
ROOT_PATH = os.getcwd()
model_path = os.path.join(ROOT_PATH, 'model')
train_data_path = os.path.join(ROOT_PATH, 'data/train_data')
test_data_path = os.path.join(ROOT_PATH, 'data/test_data')

print(f"ROOT_PATH : {ROOT_PATH}")
print(f"model_path = {model_path}")
print(f"train_data_path : {train_data_path}")
print(f"test_data_path : {test_data_path}")


# Load Train Data

## Load labels for train
df_labels = pd.read_csv('data/dirty_mnist_2nd_answer.csv')
labels = np.array(df_labels.values[:, 1:])
# print(labels) # 라벨 구성 보기

## Set Images Path
"""
.pkl file로 관리합니다.
해당 파일이 없는 경우에는 모든 데이터를 로드 한 후 file로 저장하며,
해당 파일이 있는 경우에는 pkl file을 로드 합니다.
"""

if os.path.isfile('Train_Img_path.pkl'):
    with open('Train_Img_path.pkl', 'rb') as f:
        imgs_path = pickle.load(f)
else:
    imgs_path = np.array(sorted(glob.glob(os.path.join(train_data_path, '.png'))))
    with open('Train_Img_Path.pkl', 'wb') as f:
        pickle.dump(imgs_path, f)

# print(imgs_path) # 이미지 path 보기


# Load Images
"""
구글 드라이브에서 Load 하는데 시간이 오래 걸리는 관계로, Image data를 하나의 .npy file로 관리합니다.
해당 파일이 없는 경우에는 모든 데이터를 로드한 후 file로 저징하며,
해당파일이 있는 경우에는 np.array 객체를 로드합니다.
이 방법으로 평균 2~3분 남짓한 시간으로 이미지 파일을 로드할 수 있는 장점이 있습니다.
단, file이 9GB이며 로드 시 메모리를 약 15GB 이상 사용합니다.
"""

if os.path.isfile('Imgs_numpy.npy'):
    imgs = np.load('Imgs_Numpy.npy')
    print(imgs.shape)
else:
    imgs = []
    for img_file in tqdm(imgs_path):

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
import dataloaders as dl
import models as md

# Set Train Options

## Set augmentation and transform
"""
실험을 통해 좋은 성과를 낸다고 판단된 augmentation을 train_set에 적용했습니다.
모델의 학습 환경에서 augmentation은 제외하고, transform만 동일하게 valid_set에 적용합니다.
"""
train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((331, 331)),
    T.RandomHorizontalFlip(p=0.6),
    T.RandomVerticalFlip(p=0.6),
    T.RandomRotation(40),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

valid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((331, 331)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


## Set hyper parameters
"""
모델의 크기와 resize와 같은 이유로 작은 batch size를 사용했습니다.
세션의 종료 등의 이슈로 체크포인트로부터 학습을 재개하는 경우 lr의 값을 변경할 필요가 있습니다.
"""
batch_size = 8
lr = 0.001 ### 체크 포인트에서 시작하면 이것을 변경하세요. (ex. lr = 0.001 * (0.75 ** 5))
epochs = 25
lr_scheduler_step = 5
lr_scheduler_gamma = 0.75


# Training

## Set train env
"""
여러 세션을 통해 학습을 진행할 경우, now_train_folds에 원하는 fold만 기재하여 학습할 수 있습니다.
이를 통해 여러 모델이 동시에 각 fold로 학습을 수행할 수 있습니다. 
"""
now_train_fold = [0, 1, 2, 3, 4] ### 체크 포인트에서 시작하면 이것을 변경하세요. (ex. now_train_folds = [4])
torch.cuda.empty_cache()

## Train in fold
"""
체크포인트로부터 학습을 재개하는 경우 ###로 표시된 부분을 변경할 필요가 있습니다.
체크포인트를 로드 할 수 있도록 파일 명을 기재해야 합니다. (model directory 참고)
체크포인트의 val_loss값을 valid_loss_min으로 설정해야 합니다.
체크포인트의 epoch만큼 pass 한 후 학습되도록 설정해야 합니다.

validation 수행 시 해당 epoch의 평균 loss가 계산되도록 설정해야 합니다.
valid_loss가 valid_loss_min보다 작은 경우 더 좋은 모델로 판단하고,
해당 폴드의 이전 모델을 0byte로 만들고 삭제한 후 모델의 state_dict를 저장합니다.
"""
for fold in now_train_fold :
    # Modeling
    model = md.MnistEfficientNet(in_channels=3).to(dl.device)
    # model.load_state_dict(torch.load('')) ### 만약 체크포인트에서 시작한다면 이것을 지금 가장 좋은 모델로 바꾸세요.  (ex. 'model/4fold_24epoch_0.1989_silu.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
    criterion = torch.nn.BCELoss()

    # Data
    train_idx = dl.folds[fold][0]
    valid_idx = dl.folds[fold][1]
    train_dataset = dl.MnistDataset(imgs=dl.imgs[train_idx], labels=dl.labels[train_idx], transform=train_transform)
    valid_dataset = dl.MnistDataset(imgs=dl.imgs[valid_idx], labels=dl.labels[valid_idx], transform=valid_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size//4, shuffle=False)


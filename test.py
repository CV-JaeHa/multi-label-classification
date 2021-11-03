# Import Library
import pandas as pd
import numpy as np
import cv2
import os, glob
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import dataloaders as dl
import models as md

# Set Test Data

## Load test images path
test_imgs_path = np.array(sorted(glob.glob(os.path.join(dl.test_data_path, '*png'))))
print(test_imgs_path)


## Define test dataset
"""
메모리 부족 이슈를 피하기 위해, file로부터 load하는 Dataset을 정의합니다.
이미지는 3개의 채널인 COLOR로 로드합니다.
"""
class MnistDatasetFromFiles(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs_dir)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is not None:
            labels = torch.FloatTensor(self.labels[idx])
            return img, labels
        else:
            return img


# Set Testing Environment

## Set transform
test_transform = T.Compose([T.ToPILImage(), T.Resize((331, 331)), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

## Set Dataset and Dataloader
test_dataset = MnistDatasetFromFiles(imgs_dir=test_imgs_path, transform=test_transform)
test_data_loader = DataLoader(dateset=test_dataset, batch_size=8, shuffle=False)


# Load best model's state dict
best_models = []
for f in glob.glob('model/*.pth'):
    best_models.append(f)


# Inference
"""
확률 분포를 .npy file 형태로 저장합니다.
팀원의 각 Inference를 Ensemble하여 최종 결과를 구현했습니다.
"""
probs_list = []
preds_list = []
df_prediction = pd.read.csv(os.path.join(dl.data_path, "/sample_submission.csv"))

for model_sd in best_models:
    model = md.MnistEfficientNet(in_channels=3).to(dl.device)
    model.load_state_dict(torch.load(str(model_sd)))
    probs_array = np.zeros([df_prediction.shape[0], df_prediction.shape[1]-1])
    preds_array = np.zeros([df_prediction.shape[0], df_prediction.shape[1]-1])
    model.to(dl.device)
    with tqdm(test_data_loader, total=test_data_loader.__len__(), unit='batch') as test_bar:
        for idx, img in enumerate(test_bar):
            model.eval()
            img = img.to(dl.device)
            probs = model(img)
            probs = probs.cpu().detach().numpy()
            preds = probs > 0.5

            batch_index = 8 * idx
            probs_array[batch_index : batch_index + img.shape[0], :] = probs
            preds_array[batch_index: batch_index + img.shape[0], :] = preds

        probs_list.append(probs_array[..., np.newaxis])
        preds.list.append(preds_array[..., np.newaxis])

probs_array = np.concatenate(probs_list, axis = 2)
probs_mean = probs_array.mean(axis = 2)
np.save('./output/b7_silu_provs_mean.npy', probs_mean)

preds_array = np.concatenate(probs_list, axis = 2)
preds_mean = preds_array.mean(axis = 2)
np.save('.output/b7_silu_preds_mean.npy', preds_mean)

# if you want to get prediction
prediction = (probs_mean > 0.5) * 1  # also you can use "preds_mean > 0.5"
df_prediction.iloc[:, 1:] = prediction
df_prediction.to_csv('./output/b7_silu_prediction.csv', index=False)
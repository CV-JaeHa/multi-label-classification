# Import Library
import numpy as np
import os, glob, random
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import dataloaders as dl
import models as md

#  Set random Seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
print(f'seed : {seed}')


# Set Train Options

## Set augmentation and transform
"""
실험을 통해 좋은 성과를 낸다고 판단된 augmentation을 train_set에 적용했습니다.
모델의 학습 환경에서 augmentation은 제외하고, transform만 동일하게 valid_set에 적용합니다.
"""
train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((331,331)),
    T.RandomHorizontalFlip(p=0.6),
    T.RandomVerticalFlip(p=0.6),
    T.RandomRotation(40),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

valid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((331,331)),
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
now_train_folds = [0, 1, 2, 3, 4] ### 체크 포인트에서 시작하면 이것을 변경하세요. (ex. now_train_folds = [4])
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
for fold in now_train_folds:
    # Modeling
    model = md.MnistEfficientNet(in_channels=3).to(dl.device)
    # model.load_state_dict(torch.load(''))  ### 체크포인트에서 시작할 경우 이 값을 최상의 now fold로 변경합니다. (ex. 'model/4fold_24epoch_0.1989_silu.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
    criterion = nn.BCELoss()

    # Data
    train_idx = dl.folds[fold][0]
    valid_idx = dl.folds[fold][1]
    train_dataset = dl.MnistDataset(imgs=dl.imgs[train_idx], labels=dl.labels[train_idx], transform=train_transform)
    valid_dataset = dl.MnistDataset(imgs=dl.imgs[valid_idx], labels=dl.labels[valid_idx], transform=valid_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size // 4, shuffle=False)

    valid_loss_min = 999  ### 체크포인트에서 시작할 경우 이 값을 최상의 loss로 변경합니다. (ex. valid_loss_min = 0.1989)
    for epoch in range(epochs):
        if epoch <= -1:  ### 체크포인트에서 시작할 경우 이 값을 최상의 epoch으로 변경합니다. (ex. epoch <= 24)
            continue
        # Train
        train_acc_list = []
        train_loss_list = []
        with tqdm(train_loader, total=train_loader.__len__(), unit='batch') as train_bar:
            for img, label in train_bar:
                train_bar.set_description(f'Train Epoch {epoch + 1} / {epochs}')
                X = img.to(dl.device)
                y = label.to(dl.device)

                optimizer.zero_grad()
                model.train()
                y_probs = model(X)
                print(f"input : {y_probs}, output : {y}")

                loss = criterion(y_probs, y)
                loss.backward()
                optimizer.step()

                y_probs = y_probs.cpu().detach().numpy()
                label = label.detach().numpy()
                y_preds = y_probs > 0.5
                batch_acc = (label == y_preds).mean()
                train_acc_list.append(batch_acc)
                train_acc = np.mean(train_acc_list)
                train_loss_list.append(loss.item())
                train_loss = np.mean(train_loss_list)
                train_bar.set_postfix(train_loss=train_loss, train_acc=train_acc)

        # Valid
        valid_acc_list = []
        valid_loss_list = []
        with tqdm(valid_loader, total=valid_loader.__len__(), unit='batch') as valid_bar:
            for img, label in valid_bar:
                valid_bar.set_description(f'Valid Epoch {epoch + 1} / {epochs}')
                X = img.to(dl.device)
                y = label.to(dl.device)

                optimizer.zero_grad()
                model.eval()
                y_probs = model(X)
                loss = criterion(y_probs, y)

                y_probs = y_probs.cpu().detach().numpy()
                label = label.detach().numpy()
                y_preds = y_probs > 0.5
                batch_acc = (label == y_preds).mean()
                valid_acc_list.append(batch_acc)
                valid_acc = np.mean(valid_acc_list)
                valid_loss_list.append(loss.item())
                valid_loss = np.mean(valid_loss_list)
                valid_bar.set_postfix(valid_loss=valid_loss, valid_acc=valid_acc)

        lr_scheduler.step()

        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            for f in glob.glob(os.path.join(dl.model_path, str(fold) + '*_silu.pth')):  ### 만약 다른 모델을 사용하고 싶다면 이것을 변경하세요.
                open(os.path.join(dl.model_path, f), 'w').close()
                os.remove(os.path.join(dl.model_path, f))
            torch.save(model.state_dict(), f'{dl.model_path}/{fold}fold_{epoch}epoch_{valid_loss:2.4f}_silu.pth') ### 만약 모델을 다른걸 쓰고 싶으시면 이것을 변경하세요."""
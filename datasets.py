""" 폴더 이름
1. dirty_mnist.zip라는 zip파일을 dirty_mnist라는 이름으로 압축 풀기
2. test_dirty_mnist.zip라는 zip파일을 test_dirty_mnist라는 폴더에 압축 풀기
"""

# Import Library
import torch
import numpy as np
import os, random

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

# Set Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device : {device}')
print(torch.cuda.get_device_properties(device))
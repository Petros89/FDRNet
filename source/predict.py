""" Copyright (c) 2022, Petros Apostolou.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 @Author: Petros Apostolou - apost035@umn.edu
 @Created: 6/23/2022 - 12:59 PM
 Modified: 08/12/2022 - 3:36 PM
"""


import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy
from scipy import fftpack
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import utils
import torchvision.datasets as datasets
from torch.autograd import Variable

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm

import copy
import random
import time

from torch.autograd import Variable
from utils import normalize, resample_data, smooth_data
from utils import kpss_test
from utils import arima_test
from utils import get_foot_signal
from utils import compute_feedback
from utils import get_code_list
from utils import get_code_control
from utils import get_foot_signal
from utils import generate_batches
from utils import count_parameters
from utils import calculate_accuracy
from utils import epoch_time
from model import FDRNet
from train import train_network



input_dim = 1042*16
output_dim = 1042*16

normal = pd.read_csv('../GaitData/3-3.csv')
normal_left, normal_right = get_foot_signal(["3-3", "3-4"])
normal = normalize(normal_left)
normal = normal[:1042,:].reshape(-1)

abnormal = pd.read_csv('../GaitData/1-3.csv')
abnormal_left, abnormal_right = get_foot_signal(["1-3"])
abnormal = normalize(abnormal_left)
abnormal = abnormal[:1042,:].reshape(-1)

 
normal = torch.from_numpy(normal).type(torch.Tensor)
abnormal = torch.from_numpy(abnormal).type(torch.Tensor)


model = FDRNet(input_dim, output_dim)
# load best weight to evaluate test data
model.load_state_dict(torch.load('../weights/train-FDRNet.pt'))

# use default CUDA device (GPU)
#torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model to device
#model = model.to(device)
#prediction, _ = model(abnormal, normal)
prediction, _ = model(abnormal)

# recover initial shapes
prediction = prediction.reshape((1042,16))
normal = normal.reshape((1042,16))
abnormal = abnormal.reshape((1042,16))

# copy back to host cpu
prediction = prediction.cpu().detach().numpy()
normal = normal.cpu().detach().numpy()
abnormal = abnormal.cpu().detach().numpy()

# smooth data
normal = smooth_data(normal[:,0], 0.5, 30, 6)
abnormal = smooth_data(abnormal[:,0], 0.5, 30, 6)
prediction = smooth_data(prediction[:,0], 0.5, 30, 6)

mse1 = mean_squared_error(abnormal, normal)
mse2 = mean_squared_error(prediction, normal)
print(mse1, mse2)

# plot signals
plt.title("Feedback Generation [RRZ]")
plt.plot(abnormal, label="impaired", color="red")
plt.plot(normal, label="healthy", color="blue")
plt.plot(prediction, label="feedback", color="green")
plt.legend(loc='upper right')
plt.show()

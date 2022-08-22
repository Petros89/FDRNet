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


import argparse
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
from loss import FDLoss



if __name__ == "__main__":
    
    all_codes = get_code_list()
    healthy_code, impaired_code = get_code_control(all_codes)
    print("There are {} total trials.".format(len(all_codes)))
    print("There are {} healthy trials.".format(len(healthy_code)))
    print("There are {} impaired trials.".format(len(impaired_code)))

    # let's streamline only healthy samples 
    healthy_left, healthy_right = get_foot_signal(healthy_code)
    print("Healthy Left Foot Signal Dimension: ", healthy_left.shape)
    print("Healthy Right Foot Signal Dimension: ", healthy_right.shape)
    #both = get_both_signal(healthy_code)
    
    # let's streamline only impaired samples 
    impaired_left, impaired_right = get_foot_signal(impaired_code)
    print("Impaired Left Foot Signal Dimension: ", impaired_left.shape)
    print("impaired Right Foot Signal Dimension: ", impaired_right.shape)
    
    # replace NaN with zeros
    healthy_left.replace(np.nan, 0.0)
    impaired_left.replace(np.nan, 0.0)
    healthy_right.replace(np.nan, 0.0)
    impaired_right.replace(np.nan, 0.0)
    
    # Normalize healthy data in [-1, 1]
    Hleft = normalize(healthy_left)
    Hright = normalize(healthy_right)

    # Normalize impaired data in [-1, 1]
    Ileft = normalize(impaired_left)
    Ileft = Ileft[:Hleft.shape[0]]
    Iright = normalize(impaired_right)

    # Normalize all data in [-1, 1]
    #both = normalize(both)

    # determine healthy sample dimension
    Hleft_dim = Hleft.shape[0]//len(healthy_code)
    Hright_dim = Hright.shape[0]//len(healthy_code)

    # determine healthy sample dimension
    Ileft_dim = Hleft_dim
    Iright_dim = Hright_dim

    # generate mini_batches
    Hleft_batches = generate_batches(Hleft, Hleft_dim, shuffle=False)
    Ileft_batches = generate_batches(Ileft, Ileft_dim, shuffle=False)


    print("Created {} number of healthy left foot batches.".format(len(Hleft_batches)))
    print("Created {} number of impaired left foot batches.".format(len(Ileft_batches)))
    print("healthy mini batch size: ", Hleft_batches.shape)
    print("impaured mini batch size: ", Ileft_batches.shape)


    # set reproducible parameters
    seed = 40
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ''' Skip the following 6 lines and continue in line number 154'''
    # set train and validation sample ratio
    #VALID_RATIO = 0.9
    #train_size = int(len(Hleft_batches) * VALID_RATIO)
    #valid_size = len(Hleft_batches) - train_size

    #Htrain_samples, Hvalid_samples = data.random_split(Hleft_batches,
    #                                       [train_size, valid_size])
    #Itrain_samples, Ivalid_samples = data.random_split(Ileft_batches,
    #                                       [train_size, valid_size])
   
    
    # set model's input and output dimensions
    input_dim = Hleft_batches[0].shape[0] * Hleft_batches[0].shape[1]
    output_dim = input_dim

    # convert numpy arrays to torch tensors 
    HLtrain = torch.from_numpy(Hleft_batches).type(torch.Tensor)
    ILtrain = torch.from_numpy(Ileft_batches).type(torch.Tensor)
    print("Health Tensor Size ", len(HLtrain)) 
    print("Impaired Tensor Size ", len(ILtrain)) 


    # Define Loss Function (call function in loss.py) 
    criterion = FDLoss()

    # let's free/clean GPU memory cache
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FDRNet(input_dim, output_dim)

    # allocate model and criterion to GPU device
    model.to(device)
    criterion.to(device)
    print(model)
    print("The model has {} trainable parameters.".format(count_parameters(model)))
    print("*** Start Training ***")
    print("==========================================================")
  
    
    # choose optimizer for the training loop
    optimizer = optim.SGD(model.parameters(), lr=2e-5, momentum=0.9)

    # Initialize best losses to save
    best_train_loss = float('inf')
    #best_valid_loss = float('inf')

    HLtrain = Variable(HLtrain.data, requires_grad=True)
    ILtrain = Variable(ILtrain.data, requires_grad=True)

    train_losses = []
    train_accues = []

    # Start the Training Phase
    Niters = 5
    for epoch in trange(Niters):
    
        # start time stamp
        start_time = time.monotonic()

        # compute Loss model
        train_loss = train_network(model, ILtrain, HLtrain, optimizer, criterion, device)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), '../weights/train-FDRNet.pt')

        # here can perform validation at a later stage
 
        # append trainin curves
        train_losses.append(train_loss)

        # end time stamp
        end_time = time.monotonic()
    
        # elapsed time per epoch
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch} | Train Loss: {train_loss:.8f}  |  Epoch Time: {epoch_mins}m {epoch_secs}s')


    plt.plot(train_losses, label="train_loss", color="blue")
    plt.legend()
    plt.show()

###########
### EOF ###
###########

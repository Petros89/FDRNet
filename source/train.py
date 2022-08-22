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



import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_accuracy
from sklearn.metrics import mean_squared_error
from utils import compute_feedback
import torch.optim as optim
import torch.utils.data as data
from tqdm.notebook import trange, tqdm
from torch.autograd import Variable

import copy
import random
import time
from utils import epoch_time


def train_network(model, source, target, optimizer, criterion, device):

    # initialize loss, acc
    epoch_loss = 0
    epoch_acc = 0
   
    # inform the model that is about to start training
    model.train()


    for (x, y) in zip(source, target):
        # send x,y to cuda device
        x = x.to(device)
        y = y.to(device)

        # initialize gradients
        optimizer.zero_grad()

        # get feedback prediction 
        pred, _ = model(x)

        # calculate MSE loss between prediction and target
        loss = criterion(pred, y.view(-1))
       
        # back-propagete the loss
        loss.backward()

        # update solver's step
        optimizer.step()

        # increment losses per iteration
        epoch_loss += loss.item()
      
    return epoch_loss / len(source)

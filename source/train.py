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
 
 Author: Petros Apostolou - apost035@umn.edu
 Created: 6/23/2022 - 12:59 EST (Washington DC)
 Cite: Petros Apostolou, "Gait Feedback Discovery and Correction Using Multivariate Time-Series Learning",
       PhD in Computer Science and Engineering Department, University of Minnesota, 2022.
"""
"""
Created on Sun Oct 25 11:17:43 2020
Modified on Monday June 20 10:32:23 2022
@author: Petros Apostolou | trs.apostolou@gmail.com
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


def train_network(model, source, target, optimizer, criterion, device):

    # initialize loss, acc
    epoch_loss = 0
    epoch_acc = 0
   
    # inform the model that is about to start training
    model.train()


    #for (x, y) in tqdm(zip(source, target), desc="Training", leave=False):
    #for (x, y) in tqdm(zip(source, target), total = 50, desc="Training", leave=False):
    for (x, y) in zip(source, target):
        # send x,y to cuda device
        x = x.to(device)
        y = y.to(device)


        # need to add extra variable for real diff = compute_feedback(x,y) and
        # then compute loss as MSE(pred, diff.view(-1)

        # initialize gradients
        optimizer.zero_grad()

        # get feedback prediction 
        pred, _ = model(x)


        # calculate MSE loss between prediction and target
        loss = criterion(pred, y.view(-1))

        # calculate_accuracy as loss inverse
        acc = abs(1 - loss) 

        # back-propagete the loss
        loss.backward()

        # update solver's step
        optimizer.step()

        # increment losses per iteration
        epoch_loss += loss.item()
        epoch_acc += acc.item()
      
    return epoch_loss / len(source), epoch_acc / len(source)

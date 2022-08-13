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
import torch.optim as optim
import torch.utils.data as data
from tqdm.notebook import trange, tqdm
from torch.autograd import Variable

import copy
import random
import time


def evaluate(model, source, taret, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(zip(source, target), desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x,y)

            loss = criterion(y_pred, y)

            acc = abs(1-los)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(source), epoch_acc / len(source)


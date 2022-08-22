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



class FDRNet(nn.Module):
    """Description: Feedback Recovery Network (FDRNet) is
    a novel neural network architecture to provide feedback
    recovery between two different time-series 
    """
    def __init__(self, input_dim, output_dim):
        super(FDRNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # define linear nn layers
        self.input_fc = nn.Linear(self.input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, self.output_dim)

    def forward(self, source):

        batch_size = source.shape[0]

        source = source.view(-1)

        w = F.relu(self.input_fc(source))

        h = F.relu(self.hidden_fc(w))

        pred_y = self.output_fc(h)

        return pred_y, h

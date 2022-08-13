
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
from utils import compute_feedback

class FDLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FDLoss, self).__init__()
 
    def compute_feedback(self, dat1, dat2):
        """Returns the feedback as a magnitude difference gap between
        two features. If for example featuresX, featuresY are extracted
        features, then feturesX + feedback restores featuresY.
    
        Returns
        -------
        data.type
            Data.type magnitude difference
        """
        fdback = torch.zeros(dat1.shape[0], device='cuda')
        for i in range(dat1.shape[0]):
            if dat1[i] >= 0 and dat2[i] >= 0:
                if dat1[i] > dat2[i]:
                    fdback[i] = abs(dat1[i] - dat2[i])
                else:
                    fdback[i] = -abs(dat1[i] - dat2[i])
            elif dat1[i] < 0 and dat2[i] < 0:
                if abs(dat1[i]) > abs(dat2[i]):
                    fdback[i] = abs(abs(dat1[i]) - abs(dat2[i]))
                else:
                    fdback[i] = -abs(abs(dat1[i]) - abs(dat2[i]))
            elif dat1[i] >= 0 and dat2[i] <= 0:
                fdback[i] = dat1[i] + abs(dat2[i])
            else:
                fdback[i] = -(abs(dat1[i]) + dat2[i])
        #return fdback + dat2.cuda()
        return fdback + dat2.cuda()

    def forward(self, source, target):        
        
        #inputs = F.sigmoid(inputs)       
        
        source = source.view(-1)
        target = target.view(-1)
        
        
        fdr_score = self.compute_feedback(target, source)
        #loss_score = torch.mean(fdr_score)
        loss_score = torch.mean((fdr_score - target)**2)
        #intersection = (inputs * targets).sum()                            
        #dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return loss_score 

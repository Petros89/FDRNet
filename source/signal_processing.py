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
Reference Link: https://github.com/kianData/PyTorch-Multivariate-LSTM/blob/main/torch_GitHub.py
Modified on Monday June 20 10:32:23 2022
@author: Petros Apostolou | trs.apostolou@gmail.com
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
from utils import adf_test
from utils import get_signal_period
from model import MLP
from train import train_network



if __name__ == "__main__":

# reading data frame ==================================================
normal = pd.read_csv('../GaitData/3-3.csv')
normal_left, normal_right = get_foot_signal(["3-3"])
normal = normalize(normal_left)
normal = resample_data(normal[:,0], True, 1000, 40)

abnormal = pd.read_csv('../GaitData/1-3.csv')
abnormal_left, abnormal_right = get_foot_signal(["1-3"])
abnormal = normalize(abnormal_left)
abnormal = resample_data(abnormal[:,0], True, 1000, 40)

# Filter the data, and plot both the original and filtered signals.
normal = smooth_data(normal, 0.5, 30, 6)
abnormal = smooth_data(abnormal, 0.5, 30, 6)

diff = compute_feedback(normal, abnormal)
corr = diff + abnormal


# ADF Test
### test if signal is stationary ##############
# Call the function and run the test
print(adf_test(normal))
print(adf_test(abnormal))


print("Period: ", get_signal_period(normal))

### find max freq ###
fs = 1
ft = np.fft.rfft(normal)
t = np.linspace(0, 100, 100*fs, endpoint=False)
freqs = np.fft.rfftfreq(normal.shape[0], t[1]-t[0])
mags = abs(ft)
inflection = np.diff(np.sign(np.diff(mags)))
peaks = (inflection < 0).nonzero()[0] + 1
peak = peaks[mags[peaks].argmax()]
signal_freq = freqs[peak] # Gives 0.05
print("Peak Frequency: ", signal_freq)


acf = np.correlate(normal, normal, 'full')[- normal.shape[0]:]
pdg = np.fft.rfft(acf)
freqs = np.fft.rfftfreq(normal.shape[0], t[1]-t[0])
inflection = np.diff(np.sign(np.diff(acf))) # Find the second-order differences
peaks = (inflection < 0).nonzero()[0] + 1 # Find where they are negative
delay = peaks[acf[peaks].argmax()]
signal_freq = fs/delay
print("delay ", delay)
print("signal_freq ",signal_freq)
plt.plot(freqs, abs(pdg))
plt.show()

plot_acf(normal, lags=400, color="blue")
plot_acf(abnormal, lags=400, color="red")
plt.show()

# implement arima autoregression test
test_normal = arima_test(normal)
test_abnormal = arima_test(abnormal)
print(test_normal.summary())
print(test_abnormal.summary())


# invert predictions
y_train_pred = norm_all.inverse_transform(y_train_pred.detach().numpy())
y_train = norm_all.inverse_transform(y_train.detach().numpy())
y_test_pred = norm_all.inverse_transform(y_test_pred.detach().numpy())
y_test = norm_all.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
print(y_train.shape)
trainScore = math.sqrt(mean_squared_error(y_train, y_train_pred))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test, y_test_pred))
print('Test Score: %.2f RMSE' % (testScore))

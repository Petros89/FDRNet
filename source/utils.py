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

import sys
import json
import os
from urllib.request import URLopener
import tarfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, freqz
from sklearn.utils import resample
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


DATA_HOME = "../GaitData"
CODE_LIST_FNAME = "../codes/code_list.json"


def get_filename(code):
    """Returns the filename of the signal file and the metadata file.

    Parameters
    ----------
    code : str
        Code of the trial ("Patient-Trial").

    Returns
    -------
    str
        Filename.
    """
    subject_str, trial_str = code.split("-")
    subject = int(subject_str)
    trial = int(trial_str)
    filename = os.path.join(DATA_HOME, code)
    assert os.path.exists(
        filename + ".csv"), "The code {} cannot be found in the data set.".format(code)
    return filename


def load_trial(code):
    """Returns the signal of the trial.

    Parameters
    ----------
    code : str
        Code of the trial ("Patient-Trial")

    Returns
    -------
    panda array
        Signal of the the trial, shape (n_sample, n_dimension).
    """
    fname = get_filename(code)
    df = pd.read_csv(fname + ".csv", sep=",")
    return df


def load_metadata(code):
    """Returns the metadata of the trial.

    Parameters
    ----------
    code : str
        Code of the trial ("Patient-Trial").

    Returns
    -------
    dict
        Metadata dictionary.
    """
    fname = get_filename(code)
    with open(fname + ".json", "r") as f:
        metadata = json.load(f)
    return metadata


def get_code_list():
    """Returns the list of all available codes.

    Returns
    -------
    list
        List of codes.
    """
    with open(CODE_LIST_FNAME, "r") as f:
        code_list = json.load(f)
    return code_list



def resample_data(df, mode, num_samples, random_state):
    """Returns a resampled dataset maintaining type.

    Returns
    -------
    pd.DataFrame resampled on the number of samples
        pd.DataFrame dataset.
    """
    return resample(df,
             replace=mode,
             n_samples=num_samples,
             random_state=random_state)



def normalize(data):
    """Returns the data in the scaled range [a,b].

    Returns
    -------
    np.ndarray
        Numpy array of the normalized data
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)
    

def kpss_test(data):
    print ('Results of KPSS Test:')
    kpss_score = kpss(data, regression='c', nlags="auto")
    kpss_perf = pd.Series(kpss_score[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpss_score[3].items():
        kpss_perf['Critical Value (%s)'%key] = value
    ### check if stationary ###
    return kpss_perf

def arima_test(data):
    model_arima = ARIMA(data, order = (5,1,0))
    return model_arima.fit()
    

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    a, b = butter_lowpass(cutoff, fs, order=order)
    res = lfilter(a, b, data)
    return res


def smooth_data(data, cutoff, fs, order):
    """Returns the data smoothed by a low-pass filter.

    Returns
    -------
    data.type
        Data.type smoothed with low-pass filter
    """
    return butter_lowpass_filter(data, cutoff, fs, order)



def compute_feedback(dat1, dat2):
    """Returns the feedback as a magnitude difference gap between
    two signals. If for example dataX, dataY are the 2 signals,
    then dataY + feedback restores signal dataX.

    Returns
    -------
    data.type
        Data.type magnitude difference
    """
    fdback = np.zeros(dat1.shape[0])
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
        elif dat1[i] >= 0 and dat2[i] <=0:
            fdback[i] = dat1[i] + abs(dat2[i])
        else:
            fdback[i] = -(abs(dat1[i]) + dat2[i])
    return fdback



def get_code_control(all_codes):
    """Returns the codes of different classes.

    Returns
    -------
    List
        List of codes of different classes
    """
    healthy_lst = []
    impaired_lst = []
    for code in all_codes:
        signal = load_trial(code)
        metadata = load_metadata(code)
        if metadata["PathologyGroup"] == str("Healthy"):
            healthy_lst.append(code)
        else:
            impaired_lst.append(code)
   
    return healthy_lst, impaired_lst



def get_code_steps(code):
    """Returns lists of left and right foot codes.

    Returns
    -------
    List
        Lists of left/right foot signals.
    """
    signal = load_trial(code)
    metadata = load_metadata(code)
    left_steps = np.array(metadata.pop("LeftFootActivity"))
    right_steps = np.array(metadata.pop("RightFootActivity"))
    
    return left_steps, right_steps



def get_foot_signal(control_code):
    """Returns the dictionary of left and right signals.

    Returns
    -------
    Dict
        Dict of left/right signals.
    """

    # define signal variables
    signal_ID = ["LAV", "LAX", "LAY", "LAZ", "LRV", "LRX", "LRY", "LRZ", "RAV", "RAX", "RAY", "RAZ", "RRV", "RRX", "RRY", "RRZ"]
    left = pd.DataFrame()
    right = pd.DataFrame()
    total_left = []
    total_right = []
    
    # loop over all signal variables
    for ID in signal_ID:
        flat_left = []
        flat_right = []
        left_steps = []
        right_steps = []
        left_foot_lst = []
        right_foot_lst = []
        left_total_healthy = []
        right_total_healthy = []

        # loop over all healthy samples
        for code in control_code:
            signal = load_trial(code)
            metadata = load_metadata(code)
            left_steps, right_steps = get_code_steps(code)
    
            left_foot_lst = []
            right_foot_lst = []

            # LEFT FOOT ACTIVITY
            for ls,le in left_steps:
                left_signal = signal[ID][ls:le].values.tolist()
                left_foot_lst.append(left_signal)
            flat_left = [item for sublist in left_foot_lst for item in sublist]
            left_total_healthy.append(flat_left)

            # RIGHT FOOT ACTIVITY
            for rs,re in right_steps:
                right_signal = signal[ID][rs:re].values.tolist()
                right_foot_lst.append(right_signal)
            flat_right = [item for sublist in right_foot_lst for item in sublist]
            right_total_healthy.append(flat_right)

        # get left foot total healthy dict
        new_flat_left = [item for sublist in left_total_healthy for item in sublist]
        total_left.append(new_flat_left)
        left_foot = pd.DataFrame(new_flat_left, columns=[ID])
        left = pd.concat([left, left_foot], axis=1)

        # get right foot total healthy dict
        new_flat_right = [item for sublist in right_total_healthy for item in sublist]
        total_right.append(new_flat_right)
        right_foot = pd.DataFrame(new_flat_right, columns=[ID])
        right = pd.concat([right, right_foot], axis=1)

    return left, right



def get_both_signal(control_code):
    """Returns the dictionary of total signals.

    Returns
    -------
    Dict
        Dict of total signals.
    """

    # initialize pd.df
    both = pd.DataFrame()

    # loop over all healthy samples
    for code in control_code:
        signal = load_trial(code)
        metadata = load_metadata(code)
        both = pd.concat([both, signal], axis=0)
    return both 


def generate_batches(dataset, batch_size, shuffle=True):
    """Returns dataset splitted into fixed size mini_batches.
    
    Returns
    -------
    Array
        Arrays of mini_batches of size batch_size.
    """
    n_samples = dataset.shape[0]

    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    if shuffle == True:
        np.random.shuffle(indices)

    mini_batches = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_idx = indices[start:end]
        
        if end-start < batch_size:
            break

        mini_batches.append(dataset[batch_idx])
    return np.array(mini_batches)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred.reshape(-1))).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    p_val_lim = 0.05
    sumval = 0.0
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        sumval += value
        dfoutput['Critical Value (%s)'%key] = value
    crit_val_mean = sumval/3
    if abs(dfoutput[0]) > abs(crit_val_mean) and dfoutput[1] < p_val_lim:
        print("Signal is Stationary")
    else:
        print("Signal is Non-Stationary")
    return dfoutput 


def get_signal_period(signal):
    acf = np.correlate(signal, signal, 'full')[-len(signal):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    return peaks[acf[peaks].argmax()]

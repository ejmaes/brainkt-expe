import os,sys
import shutil
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

import scipy.signal as sig 

#%% ------ BANDS
def compute_psd_bands(psds:np.array, freqs) -> np.array:
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 5.0],
                    "theta": [5.0, 8.0],
                    "alpha": [8.0, 13.0],
                    "sigma": [13.0, 16.0],
                    "beta": [16.0, 30.0]}
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    # Compute by bands
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax), :].mean(axis=1)
        X.append(psds_band.reshape(len(psds), 1, -1))
    Y = np.concatenate(X, axis=1)
    return Y # still lacking timestamps

#%% ------ Extract Spectral Power
def psd_window(data, tmin:float=0, tmax:float=None, nvps:int=1):
    """
    Compute the psd on overlapping windows to bypass TF tradeoff
    Takes about 5s for one pass over the 30min eeg

    Input:
        nvps: int, number of values per second; default 1. If > 1 overlapping computation windows
    """
    sfreq = data.info['sfreq']
    if tmax is None:
        tmax = data.get_data().shape[1] / sfreq
    if (tmax-tmin) % 1 != 0:
        tmax = tmax - ((tmax-tmin) % 1)
    fracs = np.array(range(0,nvps))/nvps
    dfs = []
    nshape = None
    t_all = None
    for x in fracs:
        # need to check the number of seconds is integer
        tmin_ = tmin + x
        tmax_ = tmax if x == 0 else (tmax-1+x) # making sure does not exceed duration of data
        t = np.arange(tmin_+0.5,tmax_+0.5)
        psds = data.copy().crop(tmin=tmin_, tmax=tmax_).compute_psd(
            fmin=0.5, fmax=45., picks='eeg', average=None, n_fft=int(sfreq)).get_data()
        # Store psd
        if nshape is None:
            nshape = psds.shape
        elif psds.shape[-1] < nshape[-1]:
            # adding 0 at the end, 
            psds = np.concatenate((psds,np.zeros((nshape[0],nshape[1],nshape[-1]-psds.shape[-1]))),axis=-1)
        elif psds.shape[-1] > nshape[-1]:
            raise IndexError(f"Shape issue - {psds.shape} should be less than {nshape}")
        dfs.append(psds)
        # Store time
        t_all = t if t_all is None else np.concatenate((t_all, t), axis=None)

    dfs = tuple([np.expand_dims(df, axis=-1) for df in dfs])
    dfs = np.concatenate(dfs, axis=-1).reshape(nshape[0], nshape[1], nshape[-1]*len(fracs))
    # removing end 0
    dfs = dfs[:,:,:t_all.shape[0]]

    return dfs, np.sort(t_all)

#%% ------ Extract at specific timepoints
#%% ------ Merge electrodes
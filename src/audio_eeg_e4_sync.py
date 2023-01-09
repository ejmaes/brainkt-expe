import os,sys
import shutil
import re
import json
import numpy as np
import pandas as pd
import audiofile
from tqdm import tqdm
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import IPython

import scipy.signal as sig 
import mne

#%% ---------- EEG
def read_EEG_from_file(eeg_path:str, ) -> mne.io.Raw:
    pass


def raw_get_data(dt:mne.io.Raw, stim_channel:str='Status', event_value:int=1, resample_fq:int=None) -> pd.DataFrame:
    """From Raw MNE data, extract data and return as dataframe. Adds Trigger location in 'marker' column
    """
    # depending on use, might want to resample from 24kHz to lower
    if resample_fq is not None:
        # resampling example: https://mne.tools/0.11/auto_examples/preprocessing/plot_resample.html
        # https://mne.tools/stable/generated/mne.io.Raw.html
        dt = dt.resample(resample_fq)
    # copy data and add bool column for markers
    dtp = dt.to_data_frame()
    markers_idx = mne.find_events(dt, stim_channel=stim_channel)
    # TODO: check in new data which events
    markers_idx = markers_idx[np.where(markers_idx[:,2] == 1)[0]]
    markers_idx = markers_idx[:,0]
    # removing other markers
    dtp['marker'] = False
    dtp.loc[markers_idx,'marker'] = True
    # returns a dataframe, not yet aligned to the audio signal
    return dtp

def _check_duration(audio_triggers:pd.DataFrame, eeg_triggers:pd.DataFrame,
            audio_time_col:str='start', eeg_time_col:str='time', dec_round:int=1, **kwargs):
    # check time difference between first and last triggers
    au = audio_triggers[audio_time_col].iloc[[0,-1]].tolist()
    eg = eeg_triggers[eeg_time_col].iloc[[0,-1]].tolist()
    au_td = np.round(au[1] - au[0], decimals=dec_round)
    eg_td = np.round(eg[1] - eg[0], decimals=dec_round)
    print(f"Durations (no rounding): audio {au[1] - au[0]} - eeg {eg[1] - eg[0]}")
    if au_td != eg_td:
        # note: round the biggest one to the smallest one
        raise ValueError(f"Please review triggers to use: duration in audio: {au_td}s - in eeg: {eg_td}s")
    # check number of triggers in each
    print(audio_triggers.shape[0], eeg_triggers.shape[0])
    # return the first triggers
    return au[0], eg[0]

def _align_eeg(audio_data:pd.DataFrame, eeg_data:pd.DataFrame, 
            audio_time_col:str='start', eeg_time_col:str='time', **kwargs):
    audio_triggers = locate_markers_all(torch.Tensor(audio_data))
    eeg_triggers = eeg_data[eeg_data.marker]
    fs = 1/(eeg_data[eeg_time_col].iloc[1] - eeg_data[eeg_time_col].iloc[0])
    audio_first_trig, eeg_first_trig = _check_duration(audio_triggers, eeg_triggers, audio_time_col, eeg_time_col, **kwargs)
    eeg_data['time_align'] = (eeg_data[eeg_time_col] - eeg_first_trig + audio_first_trig).apply(lambda x: np.round(x, int((fs // 10)+1)))
    return eeg_data[(eeg_data.time_align >= 0)] # TODO: & (eeg_data.time_align <= add dataset.audio_duration)

def get_eeg(eeg_data:pd.DataFrame, s_start:float, s_stop:float, 
            eeg_timealign_col:str='time_align', select_cols:list=None):
    if select_cols is None: 
        select_cols = eeg_data.columns
    return eeg_data[select_cols][(eeg_data[eeg_timealign_col] >= s_start) & (eeg_data[eeg_timealign_col] <= s_stop)]



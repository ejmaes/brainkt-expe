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

import scipy.signal as sig 
import sklearn.preprocessing as skp # RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from eeg_base import _load_eeg

#%% Parameters
video_path = "../data/video"
aaudio_folder = "../data/audio-aligned"
eeg_folder = "../data/eeg"
teeg_folder = "../data/eeg-aligned"
e4_folder = "../data/empatica"
markers_path = "../data/video/markers_from_video_start.csv"
markers = pd.read_csv(markers_path)

#%% ---------- Loading
def load_data(date:str, group:str, **kwargs):
    # Empty memory
    if 'data' in locals():
        del data

    vfolder = f"{date}_{group}"
    video_export = f"bkt-{date}-{group}.mov"
    vaudio_name = f"bkt-{date}-{group}.wav"
    raudio_name = f"bkt-{date}-{group}_rme.wav"
    eeg_name = f"brainkt_{date}_{group}.bdf"
    mark = markers.loc[markers.file == vfolder].iloc[0]

    # load audio from video
    vaudio, vfs = audiofile.read(os.path.join(aaudio_folder, vaudio_name))

    # load audio from RME
    raudio, rfs = audiofile.read(os.path.join(aaudio_folder, raudio_name))

    # load EEG
    data, markers_idx = _load_eeg(os.path.join(eeg_folder, eeg_name), date, group, **kwargs)

    return vaudio, vfs, raudio, rfs, data, mark, markers_idx



#%% ---------- Triggers
# Get audio triggers from signal 
def get_audio_trig(vaudio:np.array, mark:pd.Series, fs:float, 
                    spec_nfft:int=64, spec_target_fq:float=0.12500, win_side:float=0.5,
                    add_plots:bool=False, add_durations:bool=True) -> pd.DataFrame:
    audio_trig = []
    sc = { 'cam1': skp.StandardScaler(), 'cam2': skp.StandardScaler() }
    for trig in ['Start Task 1','End Task 1','End Task 2']:
        at = {'marker': trig, 'video_time': mark[trig]}
        # parameters
        audio_trigger = int(mark[trig]*fs)
        swindow = int(win_side*fs) # 1s window if win_side = 0.5
        wsr = audio_trigger - swindow
        wst = audio_trigger + swindow
        # ----- locate in signal (won't work if too noisy)
        ts = np.array(range(wsr,wst))/fs
        adf = pd.DataFrame(vaudio[:,wsr:wst].T, index=ts, columns=['cam1','cam2'])
        try: # scale - use the same scaling for all 3 markers
            check_is_fitted(sc['cam1'])
        except: # NotFittedError
            sc['cam1'].fit(vaudio[0,wsr:wst].reshape(-1, 1))
            sc['cam2'].fit(vaudio[1,wsr:wst].reshape(-1, 1))
        # scale
        adf['cam1'] = sc['cam1'].transform(adf['cam1'].to_numpy().reshape(-1, 1))
        adf['cam2'] = sc['cam2'].transform(adf['cam2'].to_numpy().reshape(-1, 1))
        # 
        at['p1-sig_time'] = adf[adf.cam1.abs() > 1.5].index.tolist()[0]
        at['p2-sig_time'] = adf[adf.cam2.abs() > 1.5].index.tolist()[0]
        # ----- locate in spectrogram
        for ch in [1,2]:
            # parameters for spectrogram: nfft = nperseg = 64 => target frequency = 0.125000
            # also issues if scaling not applied
            f, t, Sxx = sig.spectrogram(adf[f'cam{ch}'], nfft = spec_nfft, nperseg=spec_nfft) # vaudio[ch,wsr:wst]
            trig_spec = pd.DataFrame(Sxx, index=f, columns=(t+wsr)/fs)
            snd = trig_spec.loc[spec_target_fq].T 
            at[f'p{ch}-spec_time'] = snd[snd > 40].index.tolist()[0] # 0.0005 if not scaling
            #sndd = snd.diff().shift(-1)
            #sndd[sndd > 0.5].index.tolist()[0]
        # add to list
        audio_trig.append(at)
        # ----- XXX: check that audio in the signal doesn't affect (for instance, looking for _periods_ above a threshold)

    audio_trig = pd.DataFrame(audio_trig).set_index('marker')
    # Booleans
    if add_durations:
        # differences in measurements
        audio_trig['st-diff'] = audio_trig['p1-sig_time'] - audio_trig['p2-sig_time']
        audio_trig['sp-diff'] = audio_trig['p1-spec_time'] - audio_trig['p2-spec_time']
        # tasks durations
        durations = pd.concat([audio_trig.loc['End Task 1'] - audio_trig.loc['Start Task 1'], 
            audio_trig.loc['End Task 2'] - audio_trig.loc['End Task 1']], axis=1).T
        durations.index = ['Task 1', 'Task 2']
        durations.loc[:,['st-diff','sp-diff']] = np.nan
        # concatenate
        audio_trig = pd.concat([audio_trig, durations], axis=0)
    if add_plots:
        plt.clf()
        fig, ax = plt.subplots(nrows=2, figsize=(14,5), sharex=True)
        # plot soundwave
        ax[0].plot(ts, vaudio[0,wsr:wst])
        ax[0].plot(ts, vaudio[1,wsr:wst], alpha=0.5)
        # plot spectrogram
        plt.pcolormesh((t+wsr)/fs, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    return audio_trig

# Compare EEG and Audio
def check_durations(audio_trig:pd.Series, eeg_markers:list, eeg_sfreq:int, precision:float=1e-3, 
                    check_consistency:bool=False, **kwargs):
    eeg_task1 = (eeg_markers[1] - eeg_markers[0])/eeg_sfreq
    eeg_task2 = (eeg_markers[2] - eeg_markers[1])/eeg_sfreq
    dt1 = (abs(eeg_task1 - audio_trig.loc['Task 1']))
    dt2 = (abs(eeg_task2 - audio_trig.loc['Task 2']))
    # if issues raise error
    if dt1 > precision or dt2 > precision:
        raise ValueError(f'Durations are very different - check signal quality. Task 1: {dt1} - Task 2: {dt2} - Precision {precision}')
    
    # else returns padding/trim needed to align
    # note: aligning on second marker to avoid difference at the end
    val = eeg_markers[1]/eeg_sfreq - audio_trig.loc['End Task 1']
    d = {1:'trim',-1:'pad'}
    if check_consistency: # comparing columns
        return {'precision': precision, 'dt1': dt1, 'dt2': dt2}
    # otherwise return for pipeline
    print("(Precision, Task 1 Difference, Task 2 Difference): ", (precision, dt1, dt2))
    return {'pad_or_trim': d[np.sign(val)], 'duration': abs(val), 'n_': abs(int(np.ceil(val*eeg_sfreq)))}

#%% ---------- Test functions
def check_all_durations(audio_trig:pd.DataFrame, eeg_markers:list, eeg_sfreq:int, **kwargs):
    eeg_durations =  {
        'task1': (eeg_markers[1] - eeg_markers[0])/eeg_sfreq,
        'task2': (eeg_markers[2] - eeg_markers[1])/eeg_sfreq
    }
    dpt_all = {}
    for col in audio_trig.columns:
        if 'diff' not in col:
            dpt_all[col] = check_durations(audio_trig[col], markers_idx, eeg_sfreq, check_consistency=True, **kwargs)
    dpt_all = pd.DataFrame(dpt_all).T

    return eeg_durations, dpt_all
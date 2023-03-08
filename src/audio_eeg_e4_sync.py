import os,sys
import shutil
import re
import json
import numpy as np
import pandas as pd
import audiofile
from tqdm import tqdm
import datetime
import math

import seaborn as sns
import matplotlib.pyplot as plt
import IPython

import scipy.signal as sig 
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import mne

import scipy.signal as sig 
import sklearn.preprocessing as skp # RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from eeg_base import _load_eeg, trim_or_pad_eeg, add_markers, eeg_split_files, signal_quality
from e4_load_sync import read_one_watch_data, select_trig, export_retimed_e4, retime_watches_from_vid

#%% Parameters
video_path = "../data/video"
aaudio_folder = "../data/audio-aligned"
eeg_folder = "../data/eeg"
aeeg_folder = "../data/eeg-aligned-x"
e4_folder = "../data/empatica"
ae4_folder = "../data/empatica-aligned"
markers_path = "../data/video/markers_from_video_start.csv"
markers = pd.read_csv(markers_path)

manual_start_values = {
#    '221205_KMJF': [23.1958,986.2938,1975.959],
#    '221206_MMER': [50.71541950113379,1025.805,1941.3719274376415],
#    '221207_SLCB': [95.1448,1066.289387755102,2021.1558],
    '221123_MMCM': [71.4128,947.8816,1956.15]
}

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
# Audio signal Filters for Trigger location
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def take_closest_to_mark(df, col:str, thres:float, trig_time:float, trig_win:float=0.2) -> float:
    """Select the timestamp that's above a threshold and closest to the video timestamp (if later), or earliest (if anterior)
    """
    if col is None: # col is None for series => dataframe
        col = 'values'
        df = pd.DataFrame(df)
        df.columns = [col]
    df['above_thres'] = df[col].abs() > thres
    df['above_start'] = (df['above_thres']) * (df['above_thres'] != df['above_thres'].shift()) # detect new sections above threshold
    df['time_from_vtrig'] = np.abs(df.index - trig_time) 
    # remove locations furthest from trig_time
    #sel = df[df['above_thres']].index # v0
    #sel = df[df['above_thres'] & ((df.index - trig_time).abs() <= trig_win)].index # v1
    #sel = df[df['above_start'] & df['time_from_vtrig']].index # v2
    sel = df[df['above_start'] & (df['time_from_vtrig'] <= trig_win)].sort_values('time_from_vtrig').index # v3
    if len(sel) > 0:
        return sel.tolist()[0]
    return None

# Get audio triggers from signal 
def get_audio_trig_v2(vaudio:np.array, mark:pd.Series, fs:float, 
                    spec_nfft:int=64, spec_target_fq:float=0.12500, win_side:float=0.5, win_filter:float=300.,
                    scaler = skp.MinMaxScaler(feature_range=(-1,1)), 
                    add_durations:bool=True, return_all_values:bool=False, **kwargs) -> pd.DataFrame:
    """For each marker, locate the trigger. Location for several methods of detection are returned (if one fails another one might be used).

    Selection: 
    * Audio soundwave - removed
    * Audio spectrogram - both channels
    * Audio spectrogram - convoluted channels (removing noise that's only one channel)

    Soundwave is scaled between [-1,1]; spectrogram is scaled between [0,1] (minmax scaler). 
    Soundwave is scaled based on the 1rst trigger; spectrogram isn't (relative to itself)

    Markers:
    * 'Start Task 1': precedeed only by silence. Is used to fit the scaler (so that all 3 soundwaves fit into [-1,1])
    * 'End Task 1': chatter might be simultaneous. Using bandpass filter to select frequencies around trigger frequency. Then scaling.
    * 'End Task 2': may need to be padded if the video is cut short. Same as 'End Task 1'
    """
    sound_thres = 0.8
    spec_thres = 40 #0.4
    # filter: lowcut / highcut values are 
    trig_fq = 2793.825851464031 # Hz, F7 on a keyboard
    lowcut = trig_fq - win_filter
    highcut = trig_fq + win_filter
    # main function
    window_signals = {'audio':[], 'spectrogram':[]}
    audio_trig = []
    sc = { 'cam1': clone(scaler), 'cam2': clone(scaler) }
    for trig in ['Start Task 1','End Task 1','End Task 2']:
        at = {'marker': trig, 'video_time': mark[trig]}
        # parameters
        audio_trigger = int(mark[trig]*fs)
        swindow = int(win_side*fs) # 1s window if win_side = 0.5
        wsr = audio_trigger - swindow
        wst = audio_trigger + swindow
        shorta = vaudio[:,wsr:wst]
        # filtering
        shorta[0,:] = butter_bandpass_filter(shorta[0,:], lowcut, highcut, fs, order=6)
        shorta[1,:] = butter_bandpass_filter(shorta[1,:], lowcut, highcut, fs, order=6)
        # ----- locate in signal (won't work if too noisy)
        ts = np.array(range(wsr,wst))/fs
        if (trig == 'End Task 2') and (ts.shape[0] != shorta.shape[1]): # if video is cut during trig
            pad = np.zeros((2, ts.shape[0] - shorta.shape[1]))
            shorta = np.concatenate([shorta, pad], axis=1)

        adf = pd.DataFrame(shorta.T, index=ts, columns=['cam1','cam2'])
        try: # scale - use the same scaling for all 3 markers
            check_is_fitted(sc['cam1'])
        except: # NotFittedError
            sc['cam1'].fit(shorta[0].reshape(-1, 1))
            sc['cam2'].fit(shorta[1].reshape(-1, 1))
        # scale
        adf['cam1'] = sc['cam1'].transform(adf['cam1'].to_numpy().reshape(-1, 1))
        adf['cam2'] = sc['cam2'].transform(adf['cam2'].to_numpy().reshape(-1, 1))
        # Removed because often noise before / sometimes trig starts far from marker
        at['p1-sig_time'] = take_closest_to_mark(adf, 'cam1', sound_thres, mark[trig])
        at['p2-sig_time'] = take_closest_to_mark(adf, 'cam2', sound_thres, mark[trig])
        # ----- locate in spectrogram
        spec_sig = {}
        for ch in [1,2]:
            # parameters for spectrogram: nfft = nperseg = 64 => target frequency = 0.125000
            # also issues if scaling not applied
            # not adding fs in spectropgram => 0-1 must be transformed to actual range
            f, t, Sxx = sig.spectrogram(adf[f'cam{ch}'], nfft = spec_nfft, nperseg=spec_nfft) # vaudio[ch,wsr:wst]
            trig_spec = pd.DataFrame(Sxx, index=f, columns=(t+wsr)/fs)
            snd = trig_spec.loc[spec_target_fq].T 
            # scale (0-1) - and recreate Series
            #snd = pd.Series(skp.MinMaxScaler(feature_range=(0,1)).fit_transform(snd.to_numpy().reshape(-1, 1)).T[0], index=snd.index)
            spec_sig[f'p{ch}-spec_time'] = snd
            sndd = snd.diff().shift(-1) # Adding diffs but will not be used
            spec_sig[f'p{ch}-spec_time_diff'] = sndd
            at[f'p{ch}-spec_time'] = take_closest_to_mark(snd, None, spec_thres, mark[trig])
        # compute using both spectrometers - flatten where one speaks, cap the trig at 1
        spec_sig = pd.DataFrame(spec_sig)
        spec_sig['marker'] = trig
        spec_sig['video_time'] = mark[trig]
        spec_sig['cross-spec_time'] = spec_sig['p1-spec_time'] * spec_sig['p2-spec_time']
        #spec_sig['cross-spec_time'] = (spec_sig['p1-spec_time'] + spec_sig['p2-spec_time'])/2
        #spec_sig['cross-spec_time'] = skp.MinMaxScaler().fit_transform(spec_sig['cross-spec_time'].to_numpy().reshape(-1, 1))
        at[f'cross-spec_time'] = take_closest_to_mark(spec_sig, 'cross-spec_time', spec_thres, mark[trig])
        # save data
        adf['marker'] = trig
        adf['video_time'] = mark[trig]
        window_signals['audio'].append(adf)
        window_signals['spectrogram'].append(spec_sig)
        # add triggers to list
        audio_trig.append(at)
        # ----- XXX: check that audio in the signal doesn't affect (for instance, looking for _periods_ above a threshold)

    audio_trig = pd.DataFrame(audio_trig).set_index('marker')
    # Booleans
    if add_durations:
        # differences in measurements
        #audio_trig['st-diff'] = audio_trig['p1-sig_time'] - audio_trig['p2-sig_time']
        #audio_trig['sp-diff'] = audio_trig['p1-spec_time'] - audio_trig['p2-spec_time']
        # tasks durations
        durations = pd.concat([audio_trig.loc['End Task 1'] - audio_trig.loc['Start Task 1'], 
            audio_trig.loc['End Task 2'] - audio_trig.loc['End Task 1']], axis=1).T
        durations.index = ['Task 1', 'Task 2']
        #durations.loc[:,['st-diff','sp-diff']] = np.nan
        #durations.loc[:,['sp-diff']] = np.nan
        # concatenate
        audio_trig = pd.concat([audio_trig, durations], axis=0)
    # Plots: must be one outside of the function, with the best selected trigger
    # Returns
    if return_all_values:
        window_signals = {k:pd.concat(v, axis=0).reset_index(drop=False) for k,v in window_signals.items()}
        return audio_trig, window_signals
    return audio_trig

# ----- Previous version -----
"""
def take_closest_to_mark(df:pd.DataFrame, col:str, thres:float, trig_time:float,) -> float:
    #Select the timestamp that's above a threshold and closest to the video timestamp (if later), or earliest (if anterior)
    sel = df[df[col] > thres].index
    sel - trig_time
"""

# Get audio triggers from signal 
def get_audio_trig_v1(vaudio:np.array, mark:pd.Series, fs:float, 
                    spec_nfft:int=64, spec_target_fq:float=0.12500, win_side:float=0.5, win_filter:float=300.,
                    scaler = skp.MinMaxScaler(feature_range=(-1,1)),
                    add_durations:bool=True, return_all_values:bool=False) -> pd.DataFrame:
    """Simpler version. No filter to remove chatter.
    """
    # filter: lowcut / highcut values are 
    trig_fq = 2793.825851464031 # Hz, F7 on a keyboard
    lowcut = trig_fq - win_filter
    highcut = trig_fq + win_filter
    # main function
    window_signals = {'audio':[], 'spectrogram':[]}
    audio_trig = []
    sc = { 'cam1': clone(scaler), 'cam2': clone(scaler) }
    for trig in ['Start Task 1','End Task 1','End Task 2']:
        at = {'marker': trig, 'video_time': mark[trig]}
        # parameters
        audio_trigger = int(mark[trig]*fs)
        swindow = int(win_side*fs) # 1s window if win_side = 0.5
        wsr = audio_trigger - swindow
        wst = audio_trigger + swindow
        shorta = vaudio[:,wsr:wst]
        # ----- locate in signal (won't work if too noisy)
        ts = np.array(range(wsr,wst))/fs
        if (trig == 'End Task 2') and (ts.shape[0] != shorta.shape[1]): # if video is cut during trig
            pad = np.zeros((2, ts.shape[0] - shorta.shape[1]))
            shorta = np.concatenate([shorta, pad], axis=1)

        adf = pd.DataFrame(shorta.T, index=ts, columns=['cam1','cam2'])
        try: # scale - use the same scaling for all 3 markers
            check_is_fitted(sc['cam1'])
        except: # NotFittedError
            sc['cam1'].fit(shorta[0].reshape(-1, 1))
            sc['cam2'].fit(shorta[1].reshape(-1, 1))
        # scale
        adf['cam1'] = sc['cam1'].transform(adf['cam1'].to_numpy().reshape(-1, 1))
        adf['cam2'] = sc['cam2'].transform(adf['cam2'].to_numpy().reshape(-1, 1))
        # Removed because often noise before / sometimes trig starts far from marker
        #at['p1-sig_time'] = adf[adf.cam1.abs() > 1.5].index.tolist()[0]
        #at['p2-sig_time'] = adf[adf.cam2.abs() > 1.5].index.tolist()[0]
        # ----- locate in spectrogram
        spec_sig = {}
        for ch in [1,2]:
            # parameters for spectrogram: nfft = nperseg = 64 => target frequency = 0.125000
            # also issues if scaling not applied
            f, t, Sxx = sig.spectrogram(adf[f'cam{ch}'], nfft = spec_nfft, nperseg=spec_nfft) # vaudio[ch,wsr:wst]
            trig_spec = pd.DataFrame(Sxx, index=f, columns=(t+wsr)/fs)
            snd = trig_spec.loc[spec_target_fq].T 
            spec_sig[f'p{ch}-spec_time'] = snd # 0.0005 if not scaling
            sndd = snd.diff().shift(-1) # Adding diffs but will not be used
            #sndd[sndd > 0.5].index.tolist()[0]
            spec_sig[f'p{ch}-spec_time_diff'] = sndd
            if snd[snd > 40].shape[0] > 0: # 0.0005 if not scaling
                at[f'p{ch}-spec_time'] = snd[snd > 40].index.tolist()[0] # might not work, but won't throw error like that
        # compute using both spectrometers - flatten where one speaks, cap the trig at 1
        spec_sig = pd.DataFrame(spec_sig)
        spec_sig['marker'] = trig
        spec_sig['video_time'] = mark[trig]
        spec_sig['cross-spec_time'] = spec_sig['p1-spec_time'] * spec_sig['p2-spec_time']
        spec_sig['cross-spec_time'] = skp.MinMaxScaler().fit_transform(spec_sig['cross-spec_time'].to_numpy().reshape(-1, 1))
        at[f'cross-spec_time'] = spec_sig[spec_sig['cross-spec_time'] > 0.4].index.tolist()[0]
        # save data
        adf['marker'] = trig
        adf['video_time'] = mark[trig]
        window_signals['audio'].append(adf)
        window_signals['spectrogram'].append(spec_sig)
        # add triggers to list
        audio_trig.append(at)
        # ----- XXX: check that audio in the signal doesn't affect (for instance, looking for _periods_ above a threshold)

    audio_trig = pd.DataFrame(audio_trig).set_index('marker')
    # Booleans
    if add_durations:
        # differences in measurements
        #audio_trig['st-diff'] = audio_trig['p1-sig_time'] - audio_trig['p2-sig_time']
        #audio_trig['sp-diff'] = audio_trig['p1-spec_time'] - audio_trig['p2-spec_time']
        # tasks durations
        durations = pd.concat([audio_trig.loc['End Task 1'] - audio_trig.loc['Start Task 1'], 
            audio_trig.loc['End Task 2'] - audio_trig.loc['End Task 1']], axis=1).T
        durations.index = ['Task 1', 'Task 2']
        #durations.loc[:,['st-diff','sp-diff']] = np.nan
        #durations.loc[:,['sp-diff']] = np.nan
        # concatenate
        audio_trig = pd.concat([audio_trig, durations], axis=0)
    # Plots: must be one outside of the function, with the best selected trigger
    # Returns
    if return_all_values:
        window_signals = {k:pd.concat(v, axis=0).reset_index(drop=False) for k,v in window_signals.items()}
        return audio_trig, window_signals
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
    for col in audio_trig.dropna(axis=1).columns:
        dpt_all[col] = check_durations(audio_trig[col], eeg_markers, eeg_sfreq, check_consistency=True, **kwargs)

    if len(dpt_all) > 1:
        dpt_all = pd.DataFrame(dpt_all).T
        dpt_all['avg_err'] = dpt_all[['dt1','dt2']].mean(axis=1)
        dpt_all.sort_values('avg_err', inplace=True)
        return eeg_durations, dpt_all
    return eeg_durations, None

def precision_to_power(precision:float) -> float:
    return 10 ** math.ceil(math.log10(precision))

def get_best_align(audio_trig:pd.DataFrame, eeg_markers:list, eeg_sfreq:int, precision:float, prec_to_power:bool=True, **kwargs):
    eeg_durations, dpt_all = check_all_durations(audio_trig, eeg_markers, eeg_sfreq, precision=precision, **kwargs) # already sorted
    if dpt_all is None:
        raise ValueError(f"No matching could be computed for this file with precision {precision}")
    col = dpt_all.index[0]
    precision = dpt_all.iloc[0][['dt1','dt2']].max()
    if prec_to_power:   
        precision = precision_to_power(precision)
    return col, precision, eeg_durations, check_durations(audio_trig[col], eeg_markers, eeg_sfreq, precision, **kwargs), audio_trig[col] # return dpt and trigger locations in audio

#%% ---------- Plot functions
def plot_audio_trig(vaudio:np.array, fs:float, audio_trigger:float, win_side:float=0.5, # audio signal
                    spec_nfft:int=64, spec_target_fq:float=0.12500, # spectropgram results
                    savepath:str=None # saving arguments
    ):
    plt.clf()
    fig, ax = plt.subplots(nrows=3, figsize=(14,8), sharex=True)
    # plot soundwave
    swindow = int(win_side*fs) # 1s window if win_side = 0.5
    wsr = audio_trigger - swindow
    wst = audio_trigger + swindow
    ts = np.array(range(wsr,wst))/fs
    ax[0].plot(ts, vaudio[0,wsr:wst])
    ax[0].plot(ts, vaudio[1,wsr:wst], alpha=0.5)
    # plot spectrogram - interest line
    f, t, Sxx = sig.spectrogram(vaudio[0,wsr:wst], nfft = spec_nfft, nperseg=spec_nfft)
    trig_spec = pd.DataFrame(Sxx, index=f, columns=(t+wsr)/fs) # in s, t*vfs + wsr for samples index
    trig_spec.loc[spec_target_fq].T.plot(ax=ax[1])
    # plot spectrogram - full
    plt.pcolormesh((t+wsr)/fs, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)


#%% Main Pipeline
load_watches = False
overwrite_eeg = True
# vfolders can be changed depending on needs - which files have been writen already
vfolders = [vfolder for vfolder in sorted(os.listdir(video_path)) if os.path.isdir(os.path.join(video_path, vfolder))]
vfolders = list(manual_start_values.keys())#['221207_SALT', '221201_LIVS']#, '221128_EMTR', '221206_MMER']
sync_verbose = '../data/data_sync_fuse.csv'

if __name__ == '__main__':
    # For each file
    eeg_e4_align = {}
    for i, vfolder in enumerate(vfolders):
        print(f"\n----------------------\nAnalysing {i+1}/{len(vfolders)} {vfolder}", end="\t")
        add_trigger_in_pad = False
        try:
            # 0. Load Data
            [date, group] = vfolder.split('_') 
            vaudio, vfs, raudio, rfs, data, mark, markers_idx = load_data(date, group)
            if len(markers_idx) == 2:
                markers_idx = [0] + markers_idx # only one file, missing first trigger in eeg signal
                add_trigger_in_pad = True
            vfolder_data = mark.to_dict() # copy data

            # 1. Loop on audio - compute best alignment and precision
            # 1.5 if trigger locations have been computed manually, then load them instead of computing them.
            if vfolder not in manual_start_values:
                # Only using aligned video audio, but using two methods to check which is most accurate
                #for f, audio in zip(['video','rme'], [vaudio, raudio]):
                # Using first method
                eeg_res = {}
                try:
                    audio_trig = get_audio_trig_v1(vaudio, mark, vfs)
                    eeg_res[1] = get_best_align(audio_trig, markers_idx, data.info['sfreq'], precision=1e2, prec_to_power=False)
                except:
                    eeg_res[1] = [None, 1000, None, None]
                # Using second method
                try:
                    audio_trig = get_audio_trig_v2(vaudio, mark, vfs)
                    eeg_res[2] = get_best_align(audio_trig, markers_idx, data.info['sfreq'], precision=1e2, prec_to_power=False)
                except:
                    eeg_res[2] = [None, 1000, None, None]
                # Selecting best EEG alignment
                if eeg_res[1][1] == eeg_res[2][1]:
                    method = "both"
                    best_col_align, precision, eeg_durations, dpt, audio_trig_sel = eeg_res[1]
                else:
                    method = np.argmin([eeg_res[1][1], eeg_res[2][1]]) + 1
                    best_col_align, precision, eeg_durations, dpt, audio_trig_sel = eeg_res[method]
                precision = precision_to_power(precision)
            else:
                audio_trig_sel = pd.Series(manual_start_values[vfolder], index=['Start Task 1', 'End Task 1', 'End Task 2'], name='manual')
                audio_trig_sel['Task 1'] = audio_trig_sel['End Task 1'] - audio_trig_sel['Start Task 1']
                audio_trig_sel['Task 2'] = audio_trig_sel['End Task 2'] - audio_trig_sel['End Task 1']
                best_col_align = 'manual'
                method = 'manual'
                eeg_durations =  {
                    'task1': (markers_idx[1] - markers_idx[0])/data.info['sfreq'],
                    'task2': (markers_idx[2] - markers_idx[1])/data.info['sfreq']
                }
                dpt = check_durations(audio_trig_sel, markers_idx, data.info['sfreq'], precision=1e2)
                dt1 = (abs(eeg_durations['task1'] - audio_trig_sel.loc['Task 1']))
                dt2 = (abs(eeg_durations['task2'] - audio_trig_sel.loc['Task 2']))
                precision = precision_to_power((dt1+dt2)/2)
                

            # 2. Pad and trim EEG
            ndata = trim_or_pad_eeg(data, dpt, mark.loc['Stop'])
            del data
            if add_trigger_in_pad:
                trig_time = audio_trig_sel.loc['Start Task 1'] if (vfolder in manual_start_values) else audio_trig_sel.loc['Start Task 1', best_col_align]
                add_markers(ndata, trigger_time=trig_time)
            # 3. Save split EEG
            p_order = signal_quality[(signal_quality.Date == int(date)) & (signal_quality.Dyad == group)].sort_values('AdBox').Participant.tolist()
            eeg_split_files(ndata, p_order, data_path=os.path.join(aeeg_folder, f"bkt-{date}-{group}"), overwrite=overwrite_eeg)

            # 4. Load and align watches
            if load_watches:
                print("Aligning watches.")
                wfolder = os.path.join(e4_folder, vfolder)
                wfiles = sorted([x for x in sorted(os.listdir(wfolder)) if '.zip' in x])
                wfile_p1 = os.path.join(wfolder, wfiles[0])
                wfile_p2 = os.path.join(wfolder, wfiles[1])
                watch_p1 = read_one_watch_data(wfile_p1, tags_as_datetime=False)
                watch_p2 = read_one_watch_data(wfile_p2, tags_as_datetime=False)
                t1, t2 = select_trig(p1=watch_p1['tags'], p2=watch_p2['tags'], ref_time=ndata.info['meas_date'])

                retw_p1 = retime_watches_from_vid(1, watch_p1, t1, mark)
                retw_p2 = retime_watches_from_vid(2, watch_p2, t2, mark)
                export_retimed_e4(retw_p1, os.path.join(ae4_folder, f"bkt-{date}-{group}-p1.json"))
                export_retimed_e4(retw_p2, os.path.join(ae4_folder, f"bkt-{date}-{group}-p2.json"))

                vfolder_data = dict(vfolder_data, **{
                    'WTag p1': t1,  'WTag p2': t2
                })

            # 5. Save 
            vfolder_data = dict(vfolder_data, **{
                'First Watch': 'p2' if mark['Watch p2'] < mark['Watch p1'] else 'p1',
                'Video iwt': abs(mark['Watch p2'] - mark['Watch p1']),
                'EEG Start': ndata.info['meas_date'],
                'EEG Audio align': best_col_align,
                'EEG Audio precision': precision,
                'EEG Task1 Duration': eeg_durations['task1'],
                'EEG Task2 Duration': eeg_durations['task2'],
                'EEG PadOrTrim': dpt['pad_or_trim'],
                'EEG Pad Duration': dpt['duration'],
                'Align Method': method,
                'Audio Detailed ST1': audio_trig_sel['Start Task 1'],
                'Audio Detailed ET1': audio_trig_sel['End Task 1'],
                'Audio Detailed ET2': audio_trig_sel['End Task 2'],
                'Comments':''
            })
            eeg_e4_align[vfolder] = vfolder_data 
                

            # 6. Memory release
            del ndata

        except Exception as e:
            print('Issue, skipping ---- ', e)

    eeg_e4_align = pd.DataFrame(eeg_e4_align).T
    if os.path.exists(sync_verbose):
        # Load and append
        prev_data = pd.read_csv(sync_verbose, index_col=0)
        eeg_e4_align = pd.concat([prev_data, eeg_e4_align], axis=0)
    eeg_e4_align.to_csv(sync_verbose, index=True)

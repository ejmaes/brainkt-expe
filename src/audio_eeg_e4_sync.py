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
aeeg_folder = "../data/eeg-aligned"
e4_folder = "../data/empatica"
ae4_folder = "../data/empatica-aligned"
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
                    scaler = skp.MinMaxScaler(feature_range=(-1,1)),
                    add_durations:bool=True, return_all_values:bool=False) -> pd.DataFrame:
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

def get_best_align(audio_trig:pd.DataFrame, eeg_markers:list, eeg_sfreq:int, precision:float, **kwargs):
    eeg_durations, dpt_all = check_all_durations(audio_trig, eeg_markers, eeg_sfreq, precision=precision, **kwargs) # already sorted
    if dpt_all is None:
        raise ValueError(f"No matching could be computed for this file with precision {precision}")
    col = dpt_all.index[0]
    precision = dpt_all.iloc[0][['dt1','dt2']].max()
    precision = 10 ** math.ceil(math.log10(precision))
    return col, precision, eeg_durations, check_durations(audio_trig[col], eeg_markers, eeg_sfreq, precision, **kwargs) # return dpt

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
load_watches = True
overwrite_eeg = True
# vfolders can be changed depending on needs - which files have been writen already
#vfolders = [vfolder for vfolder in sorted(os.listdir(video_path)) if os.path.isdir(os.path.join(video_path, vfolder))]
vfolders = ['221123_MMCM', '221130_AMLB']#, '221128_EMTR', '221206_MMER']
sync_verbose = '../data/data_sync.csv'

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
            # Only using aligned video audio
            #for f, audio in zip(['video','rme'], [vaudio, raudio]):
            audio_trig = get_audio_trig(vaudio, mark, vfs)
            best_col_align, precision, eeg_durations, dpt = get_best_align(audio_trig, markers_idx, data.info['sfreq'], precision=1e2)

            # 2. Pad and trim EEG
            ndata = trim_or_pad_eeg(data, dpt, mark.loc['Stop'])
            if add_trigger_in_pad:
                add_markers(ndata, trigger_time=audio_trig.loc['Start Task 1', best_col_align])
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
                'EEG Pad Duration': dpt['duration']
            })
            eeg_e4_align[vfolder] = vfolder_data 
                

            # 6. Memory release
            del ndata

        except Exception as e:
            print('Issue, skipping ---- ', e)

    eeg_e4_align = pd.DataFrame(eeg_e4_align).T
    if os.path.exists(sync_verbose):
        # Load and append
        prev_data = pd.read_csv(sync_verbose)
        eeg_e4_align = pd.concat([prev_data, eeg_e4_align], axis=0)
    eeg_e4_align.to_csv(sync_verbose, index=True)

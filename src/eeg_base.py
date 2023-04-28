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

#%% Parameters
sq_path = "../data/video/cadrage-video.xlsx"
signal_quality = pd.read_excel(sq_path, sheet_name="Signal Quality")
# get bad channel dataframe
signal_quality.Impedence_High = signal_quality.Impedence_High.apply(lambda x: [] if isinstance(x,float) else ([f"{l}{n}" for l in ['A','B'] for n in range(1,33)] if x == 'all' else x.split(',')))
signal_quality.Saturating = signal_quality.Saturating.apply(lambda x: [] if isinstance(x,float) else ([f"{l}{n}" for l in ['A','B'] for n in range(1,33)] if x == 'all' else x.split(',')))
# only labelling Saturating as 'bad' channels
signal_quality['bad_channels'] = signal_quality.apply(lambda x: [f"{x.AdBox}-{ch}" for ch in x.Saturating], axis=1)
# ---- result
bad_ch = signal_quality.groupby(['Date','Dyad']).agg({'bad_channels': 'sum'})


#%% ---------- Correct original data
# Drop channels when loading EEG
# depends on which AdBox - swap 'EXG' rule
def drop_ch_rules(ch_name:str) -> bool:
    if '-C' in ch_name or '-D' in ch_name:
        return True
    elif 'EXG' in ch_name:
        if '1-' in ch_name and int(ch_name[-1]) not in [1,2,3]:
            return True
        if '2-' in ch_name and int(ch_name[-1]) not in [4,5,6]:
            return True
    elif ch_name[2:5] in ['GSR','Erg'] or ch_name[2:] in ['Resp','Plet','Temp']:
        return True 
    return False

# Keep only unique markers
def drop_markers(data, stim_channel:str='Status', repeat_window:float=5., trig_duration:float=.2, replace_value:int=0):
    """Locate markers that are within a window of the previous marker and drop them.
    """
    # 1. Locate markers
    sfreq = data.info['sfreq']
    markers_idx = mne.find_events(data, stim_channel=stim_channel)
    markers_idx = pd.DataFrame(markers_idx, columns=['idx','init_val','end_val']) # fillna with a value that won't be dropped
    markers_idx['within_window'] = (markers_idx.idx - markers_idx.idx.shift()).fillna(-repeat_window*sfreq) > repeat_window*sfreq 
    markers_idx['within_window'] = markers_idx['within_window'].cumsum()
    keep_markers = markers_idx.drop_duplicates(['init_val','end_val','within_window']).idx.tolist()
    print("Dropping markers at index: ", set(markers_idx.idx.tolist()) - set(keep_markers))
    # 2. Replace markers
    for i in keep_markers:
        event_start = i + int(trig_duration*sfreq)
        event_end = event_start + int(repeat_window*sfreq)
        data[stim_channel,event_start:event_end] = replace_value
    # Inplace, no return
    return keep_markers

def add_markers(data, trigger_time:float, stim_channel:str='Status', trig_duration:float=.2, replace_value:int=1):
    """If need arise to add trigger in EEG - after padding for instance
    """
    sfreq = data.info['sfreq']
    event_start = int(trigger_time*sfreq)
    event_end = event_start + int(trig_duration*sfreq)
    data[stim_channel,event_start:event_end] = replace_value
    # done inplace, no need to return

# Adding external triggers / annotations for analysis (IPU ends, etc)
def add_eeg_triggers(data, times:list, stim_channel:str='IPUs', replace_value:int=1):
    """Add a different channel for stim to allow for simultaneous stimuli"""
    info = mne.create_info([stim_channel], data.info['sfreq'], ['stim'])
    stim_data = np.zeros((1, len(data.times)))
    stim_data[:,[int(x*data.info['sfreq']) for x in times]] = replace_value
    stim_raw = mne.io.RawArray(stim_data, info)
    data.add_channels([stim_raw], force_update_info=True)

def add_annotations(data, df_dict:dict, start_col:str='start', stop_col:str='stop', meas_date=None, replace_annot:bool=False):
    """All dataframes will be processed at once - annotation key is dict key, df must be read from textgrid tier (start,stop)
    for instance, df_dict = {'listen': df_p2, 'speak': df_p1}
    """
    starts = []
    durs = []
    keys = []
    has_annot = len(data.annotations) > 0
    has_dt_annot = has_annot and (data.annotations.to_data_frame().onset.dtype != float)
    if has_dt_annot and (meas_date is None):
        raise AttributeError("The data has annotations, needs a meas_date argument.")
    for k,df in df_dict.items():
        #if has_dt_annot:
        #    starts.extend([x.to_datetime64() for x in df[start_col].apply(lambda x: (datetime.timedelta(seconds=x) + meas_date)).tolist()])
        #else: 
        starts.extend(df[start_col])
        durs.extend(df[stop_col] - df[start_col])
        keys.extend([k]*df.shape[0])
    annot = mne.Annotations(onset= starts, duration=durs, description=keys, orig_time=meas_date)
    if not has_annot or replace_annot:
        data.set_annotations(annot)
    else:
        data.set_annotations(data.annotations.copy() + annot)
    

#%% ---------- Loading
def _load_eeg(eeg_path:str, date:str, group:str, preload_eeg:bool=True, **kwargs):
    """Reading original files
    """
    data = mne.io.read_raw_bdf(eeg_path, preload=preload_eeg) # mne as relative
    data = data.drop_channels([x for x in data.ch_names if drop_ch_rules(x)])
    data.set_channel_types({x:'misc' for x in data.ch_names if ('-A' not in x) and ('-B' not in x) and ('Status' not in x)})
    data.info['bads'] = bad_ch.loc[(int(date),group),'bad_channels']
    #data.info['meas_date'], data.info['sfreq'], data.info['nchan']

    if preload_eeg:
        # Update marker information
        markers_idx = drop_markers(data)
    else:
        # info can be modified without loading data, markers can't
        markers_idx = None
    return data, markers_idx

def _read_participant_eeg(date:int, group:str, part:int, 
            eeg_folder:str="../data/eeg-aligned", 
            montage = mne.channels.make_standard_montage('biosemi64'), **kwargs ):
    """Read split files
    """
    eeg_name = f"bkt-{date}-{group}-p{part}-raw.fif"
    data = mne.io.read_raw_fif(os.path.join(os.path.abspath(eeg_folder), eeg_name), preload=True) 
    # channels already dropped / cropped / marked as bad earlier
    data.set_montage(montage)
    id = group[(part*2)+0: (part*2)+2] # check if [0,1] or [1,2]
    data.info['subject_info'] = {'his_id':id}
    return data

#%% ---------- Pad and Trim
def create_dummy_info(number_channels:int=10):
    ch_names = [f'CH {x+1}' for x in range(number_channels)]
    sfreq = 2048 
    info = mne.create_info(ch_names = ch_names, sfreq = sfreq)
    return info

def update_meas_date(init_date:datetime.datetime, n_samples:int, sfreq:int=2048):
    return init_date - datetime.timedelta(seconds=n_samples/sfreq)

def create_padding(n:int, info:mne.Info):
    X = np.zeros((n, info['nchan'])).T
    # ---- info can either be created (`mne.create_info`) or copied from original item since mostly identical
    # create_info only takes sfreq, ch_names, ch_types
    # meas_date can only be set using `inst.set_meas_date()`
    # ---- removing extra info, keeping what is essentially identical
    ninfo = info.copy() #copy.deepcopy(info)
    padding = mne.io.RawArray(X, ninfo)
    # ---- Update measurement date
    padding.set_meas_date(update_meas_date(info['meas_date'], n, info['sfreq']))
    return padding 
    
def trim_or_pad_eeg(data, dpt:dict, video_duration:float):
    """
    dpt: dict, result of the check_durations function
    """
    ### Pad / Trim start
    if dpt['pad_or_trim'] == 'pad':
        ndata = create_padding(dpt['n_'], data.info)
        ndata.load_data()
        ndata.append(raws=[mne.io.RawArray(data.get_data(), data.info)])
    elif dpt['pad_or_trim'] == 'trim':
        ndata = data.copy()
        ndata.crop(tmin=dpt['duration'])
        # ---- Update measurement date
        sfreq = data.info['sfreq']
        ndata.set_meas_date(ndata.info['meas_date']+ datetime.timedelta(seconds=np.ceil(dpt['duration']*sfreq)/sfreq))
    ### Pad / Trim end
    n_end = int(video_duration*data.info['sfreq'] - len(ndata))
    if n_end > 0:
        # if eeg is too short, pad the end
        enddata = create_padding(n_end+100, data.info).load_data() # adding at the end to make sure enough points are created
        ndata = mne.io.RawArray(ndata.get_data(), ndata.info) # both must be RawArray, not BDF
        ndata.append(raws=[enddata])
    ndata.crop(tmax = video_duration)
    return ndata

#%% ---------- Splitting and Writing
def eeg_split_files(data, part_names:list, data_path:str=None, 
                montage=mne.channels.make_standard_montage('biosemi64'), print_colnames:bool=False,
                overwrite:bool=False):
    """Split into one file for each participant - Rename electrodes with montage name
    """
    files = {}
    for i, part in enumerate(part_names): 
        d = data.copy()
        # Select Channels, EXG, and Status
        d.pick_channels([x for x in d.ch_names if f'{i+1}-' in x or 'Status' in x])
        # Rename using montage
        check_rename = {x: montage.ch_names[i] for i,x in enumerate(d.ch_names) if x != 'Status' and 'EXG' not in x}
        if print_colnames:
            print(check_rename)
        d.rename_channels(check_rename)
        # Saving
        files[part] = d.copy() # making sure data isn't changed in the second iteration
    
    if data_path is not None:
        for k,v in files.items():
            path = f"{data_path}-{k}-raw.fif"
            if not(os.path.exists(path)) or overwrite:
                v.save(path, overwrite=True) # creating new path
            # mne.export.export_raw('../data/test.edf', test) - format: edf, brainvision, eeglab
            # naming conventions, should finish with raw.fif or _eeg.fif.gz or ...
    else:
        return files
    

#%% Filter etc data
def add_ref(data, select='mastoids'):
    """Adding physical reference(s) as reference to the data (must be one of ['all','mastoids',1,2,3])
    This function does not add computed (avg, etc) references 
    """
    ok_list = ['all','mastoids',1,2,3]
    if select not in ok_list:
        raise ValueError(f"Adding physical references: must be one of {ok_list}, not {select}")
    ref_names = [x for x in data.ch_names if 'EXG' in x]
    if select == 'mastoids':
        ref_names = ref_names[1:]
    elif isinstance(select, int):
        ref_names = [ref_names[select-1]]
    data.set_eeg_reference(ref_channels=ref_names)

def _pre_ica_filter(data, hp:int=1, lp:int=100, notch:int=50):
    """Documentation: https://mne.tools/dev/auto_tutorials/preprocessing/30_filtering_resampling.html
    """
    data.notch_filter(freqs=[notch], picks=['eeg'])
    data.filter(l_freq=hp, h_freq=lp)

### Bridged electrodes
# From MNE tutorial https://mne.tools/stable/auto_examples/preprocessing/eeg_bridging.html
def plot_electrical_distance_matrix_os(bridged_idx, ed_matrix, cfile:str=None):
    # Electrical Distance Matrix
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    if cfile is not None:
        fig.suptitle(f'{cfile.split("/")[-1]} Electrical Distance Matrix')

    # take median across epochs, only use upper triangular, lower is NaNs
    ed_plot = np.zeros(ed_matrix.shape[1:]) * np.nan
    triu_idx = np.triu_indices(ed_plot.shape[0], 1)
    for idx0, idx1 in np.array(triu_idx).T:
        ed_plot[idx0, idx1] = np.nanmedian(ed_matrix[:, idx0, idx1])

    # plot full distribution color range
    im1 = ax1.imshow(ed_plot, aspect='auto')
    cax1 = fig.colorbar(im1, ax=ax1)
    cax1.set_label(r'Electrical Distance ($\mu$$V^2$)')

    # plot zoomed in colors
    im2 = ax2.imshow(ed_plot, aspect='auto', vmax=5)
    cax2 = fig.colorbar(im2, ax=ax2)
    cax2.set_label(r'Electrical Distance ($\mu$$V^2$)')
    for ax in (ax1, ax2):
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Channel Index')

    fig.tight_layout()
    plt.show()

def plot_distrib_edistances(ed_matrix, cfile:str=None):
    # Distribution of Electrical Distances
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    if cfile is not None:
        fig.suptitle(f'{cfile.split("/")[-1]} Electrical Distance Matrix Distribution')
    ax.hist(ed_matrix[~np.isnan(ed_matrix)], bins=np.linspace(0, 500, 51))
    ax.set_xlabel(r'Electrical Distance ($\mu$$V^2$)')
    ax.set_ylabel('Count (channel pairs for all epochs)')
    plt.show()

def list_bridged(data:dict, return_as_dict:bool=False):
    """data = {'data': raw.info, 'idx': bridged_idx, 'mat': ed_matrix}
    """
    # data = data['bkt-221128-LBRA-p1-raw.fif']
    bidx = pd.DataFrame(data['idx'], columns=['source','target'])
    bidx['source-name'] = bidx.source.apply(lambda x: data['data'].ch_names[x])
    bidx['target-name'] = bidx.target.apply(lambda x: data['data'].ch_names[x])
    bidx = bidx.groupby(['target-name'])['source-name'].agg(list)
    if return_as_dict:
        return bidx.to_dict()
    return bidx
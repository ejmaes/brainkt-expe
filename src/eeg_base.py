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
def drop_markers(data, stim_channel:str='Status', repeat_window:float=5., replace_value:int=0):
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
        event_start = i + int(0.2*sfreq)
        event_end = event_start + int(repeat_window*sfreq)
        data['Status',event_start:event_end] = replace_value
    # Inplace, no return
    return keep_markers

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
    data = mne.io.read_raw_fif(os.path.join(eeg_folder, eeg_name), preload=True) 
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
                montage=mne.channels.make_standard_montage('biosemi64'), print_colnames:bool=False):
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
            v.save(f"{data_path}-{k}-raw.fif") # creating new path
            # mne.export.export_raw('../data/test.edf', test) - format: edf, brainvision, eeglab
            # naming conventions, should finish with raw.fif or _eeg.fif.gz or ...
    else:
        return files
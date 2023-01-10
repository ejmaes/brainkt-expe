import numpy as np
import pandas as pd
import datetime
import os, sys, shutil

e4_path = "../data/empatica/"

#%% ---------- Load
# https://stackoverflow.com/questions/3451111/unzipping-files-in-python
def read_one_watch_data(watch_path, remove_once_complete:bool=True, tags_as_datetime:bool=True):
    """Reading data from Emotiv Files:
    * info.txt
    * tags.csv: each row = physical button press on the device (unix timestamp in UTC)
    * IBI.csv: inter-beat interval; time (1rst) and duration (2nd column) of the heart beat interval
    For each (other) file: 1rst row is the timestamp (UTC) of the session start; 2nd row is the sample rate (in Hz)
    * HR.csv
    * TEMP.csv
    * BVP.csv
    * EDA.csv
    * ACC.csv
    # note: HR starts after the others
    """
    # unzip if not folder
    if '.zip' in watch_path:
        shutil.unpack_archive(watch_path, watch_path[:-4])
    d = {'tags':[]}
    for file in ['HR', 'TEMP', 'BVP', 'EDA', 'ACC']:
        filepath = os.path.join(watch_path[:-4], f'{file}.csv')
        with open(filepath, 'r') as f:
            start_ts = float(f.readline().split(',')[0].replace('\n','')) # taking first for ACC
            fs = int(float(f.readline().split(',')[0].replace('\n','')))
        data = pd.read_csv(filepath, skiprows=2, header=None)
        data.columns = [file] if data.shape[1] == 1 else [f'{file}_{i}' for i in range(data.shape[1])]
        data['time'] = np.arange(0, data.shape[0]/fs, 1/fs)
        d[file] = {
            'start': start_ts, 'fs': fs, 'data': data
        }
    # adding tags
    dt_func = datetime.datetime.fromtimestamp if tags_as_datetime else float
    with open(os.path.join(watch_path[:-4], 'tags.csv'), 'r') as f:
        for ts in f.readlines():
            ts = float(ts.split(',')[0].replace('\n',''))
            d['tags'].append(dt_func(ts))
    # remove folder to preserve space / organisation
    if remove_once_complete:
        shutil.rmtree(watch_path[:-4])
    return d

def _aggregate_one_watch(one_watch_data:dict, unaligned:bool=True) -> pd.DataFrame:
    # aligning time using start_ts - unless already aligned
    if unaligned:
        starts = pd.Series({c: one_watch_data[c]['start'] for c in one_watch_data if c != 'tags'})
        min_ts = starts.min()
        starts = (starts - min_ts).to_dict()
        for c in one_watch_data:
            if c != 'tags':
                one_watch_data[c]['data']['time'] = one_watch_data[c]['data']['time'] + starts[c]
    df = pd.concat([one_watch_data[c]['data'].set_index('time') 
            for c in one_watch_data if c != 'tags'], ignore_index=False, axis=1).sort_index()
    return df.reset_index(), min_ts


#%% ----------- Triggers
# Select trigger for alignment, if more than 1
def select_trig(p1:list, p2:list, ref_time=None, compute_min:bool=True):
    """
    ref_time typically is start of video (from EEG, which fails...)
    """
    # if 0 value issue, if 1 value no brainer
    if len(p1) == 0 or len(p2) == 0:
        raise IndexError("Cannot sync, missing triggers.")
    if len(p1) == 1 and len(p2) == 1:
        return p1[0], p2[0]
    # if more than 1 value for one watch: takes the closest
    # if more than 1 value for both watches: takes the closest between the two and to a reference time (experiment start)
    if not isinstance(p1[0], float):
        raise TypeError("Dates should be in float format, not in datetime format.")
    comp = []
    for t1 in p1:
        for t2 in p2:
            comp.append({'t1':t1, 't2':t2, 'diff':abs(t2-t1)})
    comp = pd.DataFrame(comp)
    if ref_time is not None:
        if not isinstance(ref_time, float):
            ref_time = ref_time.timestamp()
        comp['time_to_ref'] = ((comp['t1']+comp['t2'])/2 - ref_time).abs()
    else:
        comp['time_to_ref'] = 0.
    if not compute_min:
        return comp
    m = comp.sort_values(['diff','time_to_ref']).iloc[0]
    return m.t1, m.t2


#%% ---- OLD - rename watches
def rename_wfiles(sessions:pd.DataFrame, watches:dict = {"A037FA":"p2", "A03CEF":"p1"}, exceptions:list = ['LKCR']):
    """After download, files are referenced by experiment time + watch name. 
    Using sessions planning and watch matching to rename.
    """
    #sessions = pd.read_excel('../data/video/cadrage-video.xlsx', sheet_name='KTANE-notes')
    #sessions['date'] = sessions['date'].apply(lambda x: x.strftime('%y%m%d'))
    #sessions['time'] = sessions['time'].apply(lambda x: x.strftime('%-Hh%M') if x.minute > 0 else x.strftime('%-Hh'))
    for f in os.listdir(e4_path):
        folder = os.path.join(e4_path, f)
        if os.path.isdir(folder) and '-' in f:
            # get dyad in sessions list
            [date, time] = f.split('-')
            m = sessions[(sessions['date'] == date) & (sessions['time'] == time)]
            if m.shape[0] != 1:
                print(f)
                raise ValueError("Should only be one in the table")
            s_name = m['group'].iloc[0]
            # rename files
            for ff in os.listdir(folder):
                file = os.path.join(folder, ff)
                watch = ff[:-4].split('_')[-1]
                part = watches[watch] 
                if s_name in exceptions: # switch
                    part = "p" + str(int((int(part[-1]) - 1.5)*(-1)+1.5))
                p_name = s_name[(int(part[-1])-1)*2:(int(part[-1])-1)*2+2]
                pattern = f"brainkt_{date}_{s_name}_{part}_{p_name}_e4_{watch}.zip"
                os.rename(file, os.path.join(folder, pattern))
                #print(ff, pattern) # controls
            # rename folder
            f_pattern = f"{date}_{s_name}"
            os.rename(folder, os.path.join(e4_path, f_pattern))
            #print(f, f_pattern, "\n") # controls
# %% [markdown]
# # Align audio between RME and videos
# 
# **Context:**
# * Videos audio isn't synced between p1 and p2
# * RME audio is sinced, but cannot be exactly aligned using DaVinci
# * RME audio _can_ be faulty
# 
# **Goal:** for each video
# 1. Load 
#     * audio from video, both channels
#     * markers file
#     * RME audio and locate clap
# 2. Compute lag between the two audios (using clap / triggers)
#     - Crosscorrelation, DTW not working (computationally to greedy), but simple substraction works
#     * Compare cam1 to RME then RME to cam2, to get cam1-cam2 alignment
# 3. For RME
#     * locate clap / markers and lag to video
#     * check that RME audio = video audio
# 4. Generate aligned audio files to use transcription on / align to EEG
# 
# **Anterior remarks**:
# * Need to check how the audio is exported in the video - whether it's 1 channel or 2


import pandas as pd
import numpy as np
import sys, os
import subprocess
import audiofile
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.signal as sig #import hilbert
import scipy.stats as ss #pearsonr
import sklearn.preprocessing as skp # RobustScaler, StandardScaler

# Parameters
video_path = "../data/video"
audio_path = "../data/audio"
markers_path = "../data/video/markers_from_video_start.csv"
markers = pd.read_csv(markers_path)

dest_folder = "../data/audio-aligned"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

#%% --------- Loading
# export audio from video
def convert_video_to_audio_ffmpeg(video_file, output_ext="wav", 
            audio_stream_to_map:int=1, target_fs:int=None, rs_use_sox:bool=False):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module

    Options: 
    * map a given stream (1: camera audio; 2: RME audio)
    * downsample: `-af aresample=resampler=soxr -ar 16000`
    """
    filename, ext = os.path.splitext(video_file)
    ffmpeg_call = ["ffmpeg", "-y", "-i", video_file]
    # stream
    if audio_stream_to_map == 2:
        filename += "_rme"
        ffmpeg_call += ["-map", f"0:{audio_stream_to_map}"]
    # resample
    if target_fs is not None:
        ffmpeg_call += ["-ar", str(target_fs)]
        if rs_use_sox: 
            ffmpeg_call += ["-af", "aresample=resampler=soxr"]
    # call
    ffmpeg_call.append(f"{filename}.{output_ext}")
    #print(' '.join(ffmpeg_call))
    subprocess.call(ffmpeg_call, 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    # return name
    return f"{filename}.{output_ext}"


#%% --------- Computing delay - Signal analysis
# Computing envelope, DTW and crosscorrelation metrics - too computationally greedy
def get_envelope(signal:np.array):
    analytic_signal = sig.hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


# Computing delay - Simpler methods
def get_delay_at_clap(vaudio:np.array, raudio:np.array, fs:int, fps:int=25, 
                    plot_delay:bool=False, return_df:bool=False, n_frames:int=2, **kwargs):
    """Look for delay (in sample points) between two signals for the same participant - window should already have been selected
    """
    if (vaudio.shape != raudio.shape):
        raise IndexError("Both signal should have the same shape.")
    elif vaudio.shape[0] > fs*10:
        raise IndexError("For higher accuracy, signal should be cut as a short window around Clap marker time.")

    # Step 1: extract envelope and normalise - window should already have been selected
    d1 = skp.MinMaxScaler().fit_transform(get_envelope(vaudio)[:, np.newaxis])
    d2 = skp.MinMaxScaler().fit_transform(get_envelope(raudio)[:, np.newaxis])
    # Step 2: look for delay
    l = []
    max_lag = int(n_frames*(fs/fps)) # delay is shorter than 1 frame, taking 2 on each side for safety - might need to change to 10
    tstart = 0 + max_lag
    tstop = d1.shape[0] - max_lag
    for step in range(-max_lag,max_lag):
        s = (d1[tstart+step:tstop+step] - d2[tstart:tstop])
        l.append([s.mean(), s.min(), s.max(), s.std()])
    l = pd.DataFrame(l, columns=['mean','min','max','std'], index=range(-max_lag,max_lag))
    
    # Step 3: return
    if plot_delay:
        l.plot()
    if return_df:
        return l
    return l['std'].idxmin()

#%% --------- Align functions
def align_two_cameras(vaudio:np.array, raudio:np.array, **kwargs):
    """Look for delay (in sample points) between two participants in video audio - window should already have been selected
    """
    if (vaudio.shape != raudio.shape):
        raise IndexError("Both signal should have the same shape.")
    elif vaudio.shape[0] != 2:
        raise IndexError("Audio should have two channels.")

    # Participant 1
    del1 = get_delay_at_clap(vaudio[0,:], raudio[0,:], **kwargs)
    # Participant 2
    del2 = get_delay_at_clap(vaudio[1,:], raudio[1,:], **kwargs)
    # Return delay p1-p2 = p1 to audio, then audio to p2
    return del1 - del2   

def pad_or_trim_audio(signal:np.array, step_shift:int, all_channels:bool=False):
    """Shift signal by step_shift, pad and trim at start/end depending on need
    Note: if all_channels=False, then only the second channel is moved
    """
    ssh = signal.shape
    step_shift = -step_shift
    if all_channels:
        new_array = []
    else:
        new_array = [signal[0,:]]
        signal = signal[1:,:]
    # set parameters
    if step_shift > 0:
        bounds = (0,step_shift)  
        sst = [step_shift,signal.shape[1]+step_shift]
    else: 
        bounds = (-step_shift,0)
        sst = [0,signal.shape[1]]
    # pad or trim each row
    for channel in range(signal.shape[0]):
        sig = signal[channel,:]
        sig = np.pad(sig, bounds, 'constant', constant_values=(0, 0))
        sig = sig[sst[0]:sst[1]]
        new_array.append(sig)

    return np.concatenate(new_array).reshape(ssh)

def align_audios(vaudio:np.array, raudio:np.array, clap_marker:float, fs:int, **kwargs):
    """Returns the aligned audios
    kwargs include fs
    """
    kwargs['fs'] = fs
    # Define marker
    audio_clap = int(mark.Clap*fs)
    swindow = int(0.5*fs)
    wstart = audio_clap - swindow
    wstop = audio_clap + swindow
    # Align two cameras
    step = align_two_cameras(vaudio[:,wstart:wstop], raudio[:,wstart:wstop], **kwargs)
    vaudio_align = pad_or_trim_audio(signal=vaudio, step_shift=step)
    # Align cameras and RME
    del1 = get_delay_at_clap(vaudio[0,wstart:wstop], raudio[0,wstart:wstop], **kwargs)
    raudio_align = pad_or_trim_audio(signal=raudio, step_shift=del1, all_channels=True)
    # Return signals
    return vaudio_align, raudio_align, (wstart,wstop), step/fs

#%% Test functions
def visual_check(vaudio_align, raudio_align, boundaries:tuple, fs:int):
    (wsr,wst) = boundaries
    ts = np.array(range(wsr,wst))/fs
    fig, ax = plt.subplots(figsize=(14,5))
    plt.plot(ts, vaudio_align[0,wsr:wst])
    plt.plot(ts, vaudio_align[1,wsr:wst], alpha=0.5)
    plt.plot(ts, raudio_align[0,wsr:wst]-0.05, alpha=0.5)
    plt.plot(ts, raudio_align[1,wsr:wst]-0.05, alpha=0.5)

# **Checking whether signal disaligns**:
# We need to check whether there is a point at which the difference between the two signals becomes stronger
def monitor_alignment(vaudio_align:np.array, raudio_align:np.array, fs:int, audio_clap:float=None, 
                plot_figure:bool=False, **kwargs):
    """Check every 10s (-ish) the alignment
    """
    df = {'ts':[], 'val':[]}
    rs = fs if audio_clap is None else int((np.round(audio_clap%1, decimals=3)+1)*fs)
    r = range(rs, vaudio_align.shape[1]-fs, 10*fs) # jump every 10 seconds until 1s before end - df won't be aligned
    for m_clap in r:
        swindow = int(0.5*fs)
        wstart = m_clap - swindow
        wstop = m_clap + swindow
        delay = get_delay_at_clap(vaudio_align[0,wstart:wstop], raudio_align[0,wstart:wstop], fs=fs, **kwargs)
        df['ts'].append(m_clap/fs)
        df['val'].append(delay/fs) # in s
    
    df = pd.DataFrame(df)
    if audio_clap is not None:
        df['ts'] = df['ts'] - audio_clap
    df = df.set_index('ts')
    if plot_figure:
        fig, ax = plt.subplots(figsize=(14,5))
        # plot evolution of disalignment
        df.plot(ax=ax)
        ax.set_xlabel("Time (in s) since recording start")
        ax.set_ylabel("Drift (in s) after aligning at clap")
        # use marker
        ax.vlines(0., df.val.min(), df.val.max(), linestyles='dashed')
    else:
        return df


#%% Main Pipeline
if __name__ == '__main__':
    vfolders = [vfolder for vfolder in sorted(os.listdir(video_path)) if os.path.isdir(os.path.join(video_path, vfolder))]

    # For each file
    drift = []
    cam_delay = {}
    overwrite_audio = False
    for i, vfolder in enumerate(vfolders):
        print(f"Analysing {i+1}/{len(vfolders)} {vfolder}", end="\t")
        try:
            # 0. Read video and create wav files
            [date, group] = vfolder.split('_')
            video = os.path.join(video_path, vfolder, f"bkt-{date}-{group}.mov")
            if overwrite_audio or not os.path.exists(video.replace('.mov','.wav')):
                vaudio_name = convert_video_to_audio_ffmpeg(video, target_fs=16000)
                raudio_name = convert_video_to_audio_ffmpeg(video, audio_stream_to_map=2, target_fs=16000)
                print("Extraction done.", end=" ")
            # get markers in audio
            mark = markers.loc[markers.file == vfolder].iloc[0]
            
            # 1. Read audio files
            # If downsampling needed: sr=[new_fs]
            vaudio, vfs = librosa.load(vaudio_name, mono=False) 
            raudio, rfs = librosa.load(raudio_name, mono=False)
            assert vfs == rfs

            # 2. Compute delay and align
            vaudio_align, raudio_align, (wsr,wst), vdiff = align_audios(vaudio, raudio, mark.Clap, fs=vfs)
            cam_delay[vfolder] = vdiff
            # 3. Compute drift and delay - nframes = 10 is a minimum 
            df = monitor_alignment(vaudio_align, raudio_align, vfs, mark['Clap'], n_frames=10).reset_index(drop=False)
            df['src'] = vfolder
            drift.append(df)
            
            # 4. Write realigned audio to file
            print("Writing audio...", end=" ")
            output_file = os.path.join(dest_folder, f"bkt-{date}-{group}.wav")
            audiofile.write(output_file, vaudio_align, vfs, bit_depth=32)
            audiofile.write(output_file.replace('.wav','_rme.wav'), raudio_align, vfs, bit_depth=32)
            print('Done.')
        except Exception as e:
            print(e)

    # Analysing
    # Are there sudden jumps in the values for some folders? How frequently?

    drift = pd.concat(drift, axis=0)
    g = sns.relplot(data=drift, x='ts',y='val', col='src', kind="line", col_wrap=3)
    for ax in g.axes:
        ax.axvline(x=0, linestyle='dashed')
    plt.plot()

    # save delay data


# %% [markdown]
# Objectives: 
# * Check slope (drift) is the same for all
# * Locate and list all skips ( > 0: RME deletes audio ; < 0: camera deletes audio)

# %%
# Method 1 - transform ts into datetime index
# pd.to_datetime(drift[(drift.src == '221208_MBLB') & (drift.ts > 0)].ts.tolist(), unit="s")
# Method 2 - transform ts into period index
#pd.PeriodIndex(drift[(drift.src == '221208_MBLB') & (drift.ts > 0)].ts.apply(
#    lambda x: f"00:{str(int(x//60)).zfill(2)}:{str(int(x%60)).zfill(2)}"), freq="S")
# NOT WORKING FOR SECONDS
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(series, model='additive')




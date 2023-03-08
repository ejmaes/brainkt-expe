#!/usr/bin/python
"""
@author: Eliot Maës
@creation: 2023/01/23

Goal: creating alignment textgrid using ipus, transcription done by whisper on server (csv files), eventual alignment

Note: alignment can be done either with whisper or whisperx (which adds word-level alignment)

To reset after error: find . -name "bkt-221*_mono*.wav" -delete 

"""
import os,sys
import re
import numpy as np
import pandas as pd
import audiofile
from tqdm import tqdm
from datetime import datetime
from glob import glob
import textgrid
import shutil
import argparse

import torch
import whisper
import whisperx
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_PAUSE = 0.3

# Local functions
from bas_pipeline_analysis import AlignTranscription
from textgrid_utils import read_tier, write_tier
from whisper_asr import AudioFileDataset

# Parameters
audio_folder = '/Users/eliot/Downloads/transcr/audio' #'../data/audio-aligned/video'
transcript_folder = '/Users/eliot/Downloads/transcr/audio' #'../data/transcript'
audio_files = sorted([x for x in os.listdir(audio_folder) if ('.wav' in x) and ('mono' not in x)])
audio_files = ['bkt-221207-LBMC.wav', 'bkt-221207-SALT.wav']
dir_files = sorted([os.path.join(transcript_folder, x) for x in os.listdir(transcript_folder)])
transcript_files = sorted([x for x in dir_files if '.csv' in x])

#%% ------ Cleanup and write tiers
def drop_dup_rows(df:pd.DataFrame) -> pd.DataFrame:
    """In some weird occasions, two transcriptions on the same spot.
    Need to check: whether to remove only one, to remove the shortest one, drop an 0 duration row, or move boundaries
    Should be run from first load of the df bc issues in words concatenation too
    """
    ors = df.shape[0]
    # Drop empty rows
    df = df[(df.stop - df.start) > 0].reset_index(drop=True)
    # Locate overlaps
    overlaps = ((df['start'] - df['stop'].shift()).fillna(1.) < 0)
    print('nb overlaps:', overlaps.sum()) # if overlaps, then crash ==> realign necessary
    if overlaps.sum() > 0:
        for idx, _ in df[overlaps].iterrows():
            print(df.loc[idx-4:idx+4, ['text','start','stop']])
            if idx not in df.index: # might have been deleted already
                continue
            elif df.loc[idx-1,'stop'] < df.loc[idx,'stop']: # move boundaries
                df.loc[idx,'start'] = df.loc[idx-1,'stop']
            else: 
                prow_dur = df.loc[idx-1,'stop'] - df.loc[idx-1,'start'] 
                row_dur = df.loc[idx,'stop'] - df.loc[idx,'start']
                if prow_dur < row_dur and df.loc[idx-1,'start'] <= df.loc[idx,'start']: # remove prev row
                    df.drop([idx-1], inplace=True)
                else:
                    i = 1
                    while df.loc[idx-1,'stop'] > df.loc[idx+i,'start']: # compute number of next rows to drop
                        i+=1
                    df.drop(list(range(idx, idx+i)), inplace=True)
    print(f'Dropped {ors - df.shape[0]} rows - overlaps')
    return df.reset_index(drop=True)


def cw_tier(df:pd.DataFrame, **write_kwargs):
    df['start'] = df['start'].apply(lambda x: np.round(x,3))
    df['stop'] = df['stop'].apply(lambda x: np.round(x,3)) # issues if no rounding
    # overlaps were taken care of earlier
    #if overlaps.sum() > 0:
        # realign, this isn't done in the write_tier function
    #    df = df[~overlaps]
    #    df.reset_index(drop=True, inplace=True)
    print(f"Writing tier {write_kwargs['annot_tier']}")
    write_tier(df, **write_kwargs)

#%% ------ Handling transcriptions
def load_csv(path:str, max_duration:float, fs_ratio:float):
    """Loading transcription from whisper / whisperx and cleaning"""
    df = pd.read_csv(path, keep_default_na=False, na_values=['']).rename(columns={'end':'stop'})
    df = df[~df.text.isin(['...','.'])].reset_index(drop=True) # removing empty sentences
    # rounding start / stop
    df.start = df.start.apply(lambda x: np.round(x,decimals=3))
    df.stop = df.stop.apply(lambda x: np.round(x,decimals=3))
    # duration issues if audio wasn't loaded properly into model
    #if (max_duration+5) < df['stop'].iloc[-1]: # might have a few more ms for random reasons
    #    df['stop'] = df['stop']*fs_ratio
    #    df['start'] = df['start']*fs_ratio
    #    print("Audio rate was not properly passed to transcription model, recomputing bounds.")
    # drop rows that overlap
    df = drop_dup_rows(df)
    return df

def wti(sipu:pd.DataFrame, words:pd.DataFrame):
    """
    Associates words from the words df (whisperx) to the ipu number (sppas)
    Note: all words must be in _exactly_ one ipu
    Empty rows of words have already been filtered out
    """
    filled_ipu = []

    # option 2 - associate word by word, by computing which ipu the word is longer in
    words['ipu'] = None
    for idx, word in words.iterrows():
        word_ipus = sipu[(word.stop > sipu.start) | (word.start < sipu.stop)]
        if (word.stop - word.start) == 0:
            print(f"Issue line {idx} word '{word}': duration is 0.")
        elif word_ipus.shape[0] > 0:
            if word_ipus.shape[0] > 1:
                word_ipus['ratio'] = sipu.apply(lambda x: 
                            (min(word.stop, x.stop) - max(word.start, x.start)) / (word.stop - word.start), axis=1)
                word_ipus.sort_values(by='ratio', inplace=True, ascending=False)
            words.loc[idx, 'ipu'] = word_ipus.index[0]
        else:
            print(f"Issue line {idx} word '{word}': matching no ipu.")
    
    for idx, ipu in sipu.iterrows():
        # option 1 - IPU by IPU, not caring about duplicates
        #ipu_words = words[(words.stop > ipu.start) & (words.start < ipu.stop)]
        # option 2 -
        ipu_words = words[words.ipu == idx]
        if ipu_words.shape[0] > 0:
            try:
                filled_ipu.append({
                    'start': ipu.start, 'stop': ipu.stop,
                    'text': ' '.join(ipu_words.text.tolist())
                })
            except Exception as e:
                print(ipu_words[['text','start','stop']])
                raise e
    
    filled_ipu = pd.DataFrame(filled_ipu)
    return filled_ipu

#%% ------ Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_bas_alignment", '-a', action="store_true", help="Whether to do BAS alignment")
    parser.add_argument("--include_all_tiers", '-t', action="store_true", help="Whether to include all tiers in output transcription")
    parser.add_argument("--overwrite", '-o', action="store_true", help="Whether to recompute everything for completed files")
    
    args = parser.parse_args()
    return args

#%% ------ Main
if __name__ == '__main__':
    args = parse_arguments()
    write_kwargs = {'text_column': 'text', 'timestart_col': 'start', 'timestop_col': 'stop'}
    max_nb_tiers = 2 + 2*3*args.include_all_tiers

    # audio and csv files have the same patterns
    for afile in audio_files:
        print(afile.split('-')) 
        _, _, group = afile.split('-') # should be ok except algorithm restart
        groups = [group[0:2], group[2:4]]
        audio_file = os.path.join(audio_folder, afile)
        dataset = AudioFileDataset(audio_file, ) 
        write_kwargs['file_duration'] = dataset.audio_duration
        write_kwargs['file_name'] = os.path.join(transcript_folder, afile.replace('.wav','.TextGrid'))

        # Check if skip
        if write_kwargs['file_name'] in dir_files:
            try:
                completed_file = (len(textgrid.TextGrid.fromFile(write_kwargs['file_name'])) >= max_nb_tiers )
                if completed_file and not args.overwrite: 
                    print("Already analysed - skipping")
                    continue
                elif not completed_file or args.overwrite:
                    print("Existing partial file - removing")
                    os.remove(write_kwargs['file_name'])
            except AttributeError:
                print("Issue with file, dropping and recomputing")
                os.remove(write_kwargs['file_name'])

        # creates IPU files then deletes it => dataset.ipus_bounds, with a 'channel' col
        # actually not deleted but at least no need to reload
        ipus = dataset.ipus_bounds
        # might have not be read correctly
        try:
            for ch in range(dataset.nb_channels):
                speaker = groups[ch]
                sipu = ipus[ipus['channel'] == ch]

                # two transcripts
                tr_file = os.path.join(transcript_folder, afile.replace('.wav',f'_{ch}.csv'))
                if tr_file not in transcript_files:
                    break
                
                # check if alignment exists as csv
                al_file = tr_file.replace(f'_{ch}.csv',f'_{ch}-word.csv')
                al_file = al_file if al_file in transcript_files else None
                # case 1: just log those into ipus
                if al_file is not None:
                    words = load_csv(al_file, dataset.audio_duration, (dataset.fs/dataset.or_fs))

                    filled_ipu = wti(sipu, words)
                    # write computed tier
                    write_kwargs['annot_tier'] = f'aligned-{speaker}'
                    cw_tier(filled_ipu, **write_kwargs)

                if args.include_all_tiers or (al_file is None):
                    # Original transcription
                    sent = load_csv(tr_file, dataset.audio_duration, (dataset.fs/dataset.or_fs))
                    write_kwargs['annot_tier'] = f'sent-{speaker}'
                    cw_tier(sent, **write_kwargs)
                    # IPUs
                    write_kwargs['annot_tier'] = f'ipu-{speaker}'
                    cw_tier(sipu, **write_kwargs)
                if args.include_all_tiers:
                    # Words
                    write_kwargs['annot_tier'] = f'words-{speaker}'
                    cw_tier(words, **write_kwargs)

                # case 2: run bas alignment on ipus
                # case 3: run bas alignment on original transcript
                if args.do_bas_alignment:
                    transcription_tier = f'sent-{speaker}' if (al_file is None) else f'aligned-{speaker}'
                    # run BAS Alignment 
                    ns_path = os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{ch}.wav")
                    ts_path = write_kwargs['file_name']
                    at = AlignTranscription(audio_path=ns_path, transcription_path=ts_path, 
                        transcription_tier=transcription_tier, lg='fra-FR')
                    at.run_pipeline(compress=True)

                    if transcription_tier == f'sent-{speaker}':
                        # parse bas into ipus
                        #outfile = at.aligned_transcription.replace('bas','bas-ipus')
                        # read file
                        words = read_tier(at.aligned_transcription, tier_name="ORT-MAU")
                        # create ipus - columns start, stop
                        words['pause_duration'] = (words['start'] - words['stop'].shift()).fillna(0.) > MIN_PAUSE
                        words['ipu_id'] = words['pause_duration'].cumsum()
                        ipus = words.groupby('ipu_id').agg({
                            'start': 'min', 'stop': 'max', 'text': lambda x: ' '.join(list(x))
                        })
                        # write ipus
                        write_tier(ipus, **write_kwargs)#outfile)

        except Exception as e:
            print(f"Issue with file {afile}: {e}")


        #audios = {speaker: os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{speaker}.wav") for speaker in range(dataset.nb_channels)}
        #textgrids = {speaker: os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{speaker}.TextGrid") for speaker in range(dataset.nb_channels)}

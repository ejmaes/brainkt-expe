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

# Local functions
from bas_pipeline_analysis import AlignTranscription
from textgrid_utils import read_tier, write_tier
from whisper_asr import AudioFileDataset

# Parameters
audio_folder = '../data/audio-aligned/video'
transcript_folder = '../data/transcript'
audio_files = [x for x in os.listdir(audio_folder) if '.wav' in x]
transcript_files = [os.path.join(transcript_folder, x) for x in os.listdir(transcript_folder)]

#%% ------ Cleanup and write tiers
def cw_tier(df:pd.DataFrame, **write_kwargs):
    df.start = df.start.apply(lambda x: np.round(x,2))
    df.stop = df.stop.apply(lambda x: np.round(x,2)) # issues if no rounding
    overlaps = ((df['start'] - df['stop'].shift()).dropna() < 0)
    print('nb overlaps:', overlaps.sum())
    write_tier(df, **write_kwargs)


#%% ------ Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_bas_alignment", '-a', action="store_true", help="Whether to do BAS alignment")
    
    args = parser.parse_args()
    return args

#%% ------ Main
if __name__ == '__main__':
    args = parse_arguments()
    write_kwargs = {'text_column': 'text', 'timestart_col': 'start', 'timestop_col': 'stop'}

    # audio and csv files have the same patterns
    for afile in sorted(audio_files):
        print(afile.split('-')) 
        _, _, group = afile.split('-') # should be ok except algorithm restart
        groups = [group[0:2], group[2:4]]
        audio_file = os.path.join(audio_folder, afile)
        dataset = AudioFileDataset(audio_file, ) 
        write_kwargs['file_duration'] = dataset.audio_duration
        write_kwargs['file_name'] = os.path.join(transcript_folder, afile.replace('.wav','.TextGrid'))

        # creates IPU files then deletes it => dataset.ipus_bounds, with a 'channel' col
        # actually not deleted but at least no need to reload
        ipus = dataset.ipus_bounds
        # might have not be read correctly
        try:
            for ch in range(dataset.nb_channels):
                speaker = groups[ch]

                # two transcripts
                tr_file = os.path.join(transcript_folder, afile.replace('.wav',f'_{ch}.csv'))
                if tr_file not in transcript_files:
                    break
                sent = pd.read_csv(tr_file).rename(columns={'end':'stop'})
                if dataset.audio_duration < sent['stop'].iloc[-1]:
                    sent['stop'] = sent['stop']*(dataset.fs/dataset.or_fs)
                    sent['start'] = sent['start']*(dataset.fs/dataset.or_fs)
                write_kwargs['annot_tier'] = f'sent-{speaker}'
                cw_tier(sent, **write_kwargs)
                
                # check if alignment exists as csv
                al_file = tr_file.replace(f'_{ch}.csv',f'_{ch}-word.csv')
                al_file = al_file if al_file in transcript_files else None
                if al_file is not None or args.do_bas_alignment:
                    if al_file is not None:
                        words = pd.read_csv(tr_file).rename(columns={'end':'stop'})
                        if dataset.audio_duration < words['stop'].iloc[-1]:
                            words['stop'] = words['stop']*(dataset.fs/dataset.or_fs)
                            words['start'] = words['start']*(dataset.fs/dataset.or_fs)
                    else:
                        # run BAS Alignment 
                        ns_path = os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{ch}.wav")
                        ts_path = write_kwargs['file_name']
                        at = AlignTranscription(audio_path=ns_path, transcription_path=ts_path, 
                            transcription_tier=f"spk{speaker}", lg=args.language)
                        at.run_pipeline(compress=True)

                        # parse bas into ipus
                        outfile = at.aligned_transcription.replace('bas','bas-ipus')
                        # read file
                        words = read_tier(at.aligned_transcription, tier_name="ORT-MAU")
                        # create ipus - columns start, stop
                        words['pause_duration'] = (words['start'] - words['stop'].shift()).fillna(0.) > 0.3
                        words['ipu_id'] = words['pause_duration'].cumsum()
                        ipus = words.groupby('ipu_id').agg({
                            'start': 'min', 'stop': 'max', 'text': lambda x: ' '.join(list(x))
                        })
                        # write ipus
                        write_tier(ipus, outfile)

                    # write alignment
                    write_kwargs['annot_tier'] = f'words-{speaker}'
                    cw_tier(words, **write_kwargs)

                # IPUs
                sipu = ipus[ipus['channel'] == ch]
                write_kwargs['annot_tier'] = f'ipu-{speaker}'
                cw_tier(sipu, **write_kwargs)
        except:
            pass


        #audios = {speaker: os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{speaker}.wav") for speaker in range(dataset.nb_channels)}
        #textgrids = {speaker: os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{speaker}.TextGrid") for speaker in range(dataset.nb_channels)}
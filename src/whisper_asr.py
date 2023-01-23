#!/usr/bin/python
"""
@author: Eliot Maës
@creation: 2022/11/11

Transcription using OpenAI Whisper
Documentation on how to install Whisper is on the [OpenAI GitHub Page](https://github.com/openai/whisper) | [Colab example](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2)

------
Installing Whisper

#!sudo apt install ffmpeg mediainfo sox libsox-fmt-mp3
!brew install ffmpeg
!brew install sox
!brew install mediainfo
!pip install audiofile
!pip install setuptools-rust
!pip install git+https://github.com/openai/whisper.git 

------
CLI use of the model

`ffmpeg` can be used to [extract audio from the video](https://stackoverflow.com/questions/9913032/how-can-i-extract-audio-from-video-with-ffmpeg): 
```
ffmpeg -i ~/Downloads/tmp/bkt-pilot-221103.mov -ss 00:00:00 -t 00:00:45.0 -q:a 0 -map a ~/Downloads/tmp/1minaudio.mp3
```

In our case, `whisper` can be called from the command line, with the output:
```bash
(base) eliot@Eliots-MBP aclpubcheck % whisper ~/Downloads/tmp/1minaudio.mp3 --language French
100%|████████████████████████████████████████| 461M/461M [09:57<00:00, 809kiB/s]
/Users/eliot/miniconda3/lib/python3.9/site-packages/whisper/transcribe.py:78: UserWarning: FP16 is not supported on CPU; using FP32 instead
  warnings.warn("FP16 is not supported on CPU; using FP32 instead")
[00:00.000 --> 00:13.320]  Donc qu'est ce que tu as comme module comme petit carré? Il y a une horloge à côté de l'horloge
[00:13.320 --> 00:24.840]  et il y a une espèce de balise avec un chronomètre et il y a un bouton annulé il y a quatre
[00:24.840 --> 00:30.840]  boutons rouge bleu vert et jeune rouge bleu vert jeune on va aller là dessus je clique sur cela
[00:30.840 --> 00:41.400]  alors ça s'appelle un saimon saimon 16 donc un des quatre boutons va s'allumer oui voilà quel
[00:41.400 --> 01:11.400]  bouton s'allumer et jeune dans ce cas
```

Downsides:
* Loose audio boundaries
* No distinctions between channels

Steps need to be taken to split between channels and align the transcription to IPUs/Words.

------
General Information:
# available models: tiny base small medium large
# default setting from command line: using 'small' model

"""
import os,sys
import re
import json
import numpy as np
import pandas as pd
import audiofile
from tqdm import tqdm
from datetime import datetime
from glob import glob
import textgrid
import shutil
import argparse

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# SPPAS
SPPAS_PATH = "/Users/eliot/Documents/tools/SPPAS"
sys.path.append(SPPAS_PATH)
# reading / writing textgrids
import sppas.src.anndata.aio.readwrite as spp
import sppas.src.anndata as sad
# searching for IPUs
import sppas
from sppas.src.annotations import sppasParam, sppasAnnotationsManager
from sppas.src.plugins import sppasPluginsManager

# Whisper
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local functions
from bas_pipeline_analysis import AlignTranscription
from textgrid_utils import read_tier, write_tier

# %% File Parsing
def get_mel(audio, device=DEVICE):
    audio = whisper.pad_or_trim(audio.flatten()).to(device) 
    # docstring: whisper.pad_or_trim(array, length: int = 480000, *, axis: int = -1)
    mel = whisper.log_mel_spectrogram(audio)
    return mel

def resample_audio(audio, orig_fs:int, target_fq:int=16000, device=DEVICE):
    if not isinstance(audio, torch.Tensor):
        audio = torch.Tensor(audio, device=device)
    # resample
    rs = torchaudio.transforms.Resample(orig_freq=orig_fs, new_freq=target_fq)
    return rs(audio)

class AudioFileDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, device=DEVICE, fs_th:int=16000, cleanup_after_splitting:bool=False):
        self.filepath = filepath if filepath[0] != '~' else os.path.expanduser(filepath)
        self.audio_duration = audiofile.duration(filepath)
        self.audio, self.fs = audiofile.read(filepath, always_2d=True)
        self.nb_channels = audiofile.channels(filepath)
        self.save_folder = os.path.realpath(os.path.dirname(self.filepath))
        self.filename = self.filepath.split('/')[-1]
        self.device = device
        self.cleanup_onechannel = cleanup_after_splitting
        print(f"File {filepath}: {self.nb_channels} channels, sampling frequency = {self.fs}Hz, {self.audio_duration}s")
        print('Creating dataset...', end=' ')
        if self.fs != fs_th:
            self._resample(target_fq=fs_th)
        self.ipus_bounds = self._searchipus()
        self._create_dataset()
        self.datasets = {'base':self.dataset}
        print('done.')

    def _resample(self, target_fq:int=16000):
        # to tensor + resample
        self.or_audio = self.audio
        self.audio = resample_audio(self.audio, self.fs, target_fq, device=self.device)
        self.or_fs = self.fs
        self.fs = target_fq

    def _searchipus(self):
        """For each channel, locate IPUs using SPPAS. 
        1. SPPAS creates new files 
        2. read from those files then delete them
        """
        # SPPAS - activate
        spass_log=f"searchipus-{datetime.now().strftime('%y%m%d-%H:%M:%S')}-{self.filepath}"
        actions = ['searchipus']
        parameters = sppasParam([f"{x}.json" for x in actions])
        for x in actions:
            ann_step_idx = parameters.activate_annotation(x)
            ann_options = parameters.get_options(ann_step_idx)

        # Files - for each channel
        for c in range(self.nb_channels):
            # extract sound and create a new file
            ns = self.audio[c,:]
            ns_path = os.path.join(self.save_folder, f"{self.filename[:-4]}_mono_{c}.wav")
            audiofile.write(ns_path, ns, self.fs)
            parameters.add_to_workspace(ns_path)
    
        # SPPAS - Fix the output file extension and others
        parameters.set_lang("fra")
        parameters.set_output_extension('.TextGrid', "ANNOT")
        parameters.set_report_filename(spass_log)
        # SPPAS - Execute pipeline
        process = sppasAnnotationsManager()
        process.annotate(parameters)

        # Files - read each file to dataset
        ipus_bounds = []
        for c in range(self.nb_channels):
            # use relpath to get the relative path from SPPAS to target folder
            # uses os.getcwd() to compute path if given in relative - os.path.expanduser() necessited
            #self.save_folder = os.path.relpath(os.path.dirname(self.filepath), start=SPPAS_PATH)
            ns_path = os.path.join(self.save_folder, f"{self.filename[:-4]}_mono_{c}.wav")
            tg_path = f'{ns_path[:-4]}-ipus.TextGrid'
            tg = textgrid.TextGrid.fromFile(tg_path)
            for t in tg[0]:
                if t.mark not in ["#",""]: # check that
                    ipus_bounds.append({
                        'channel': c, 'start': t.minTime, 'stop': t.maxTime, 'text': 'ipu'
                    })
            # Files - cleanup sound and textgrid
            if self.cleanup_onechannel:
                os.remove(ns_path)
                os.remove(tg_path)
        
        return pd.DataFrame(ipus_bounds)

    def _create_dataset(self, fs:int=16000): # fs set by whisper
        # from bounds + signal + fs, extract signal for each ipu
        dataset = {}
        for idx, row in tqdm(self.ipus_bounds.iterrows()):
            dataset[idx] = [row.start, row.stop, int(row.channel), self.audio[int(row.channel), int(row.start*fs):int(row.stop*fs)]]
        self.dataset = dataset

    def _create_long_dataset(self, maximum_pause_duration:float=3, fs:int=16000, **kwargs):
        self.ipus_bounds['duration'] = self.ipus_bounds.stop - self.ipus_bounds.start
        self.ipus_bounds['pause'] = (self.ipus_bounds.start.shift(-1) - self.ipus_bounds.stop).fillna(0.) # last value
        # aggregate - need to keep it shorter than 300s
        self.ipus_bounds['new_line'] = self.ipus_bounds['pause'] > maximum_pause_duration
        self.ipus_bounds['new_line_idx'] = self.ipus_bounds['new_line'].cumsum()
        ipu_concat = self.ipus_bounds.groupby(['channel','new_line_idx']).agg({ 'start': 'min', 'stop':'max' }).reset_index(drop=False)
        # call cataset - in other variable
        dataset = {}
        for idx, row in tqdm(ipu_concat.iterrows()):
            dataset[idx] = [row.start, row.stop, int(row.channel), self.audio[int(row.channel), int(row.start*fs):int(row.stop*fs)]]
        self.datasets[f'long_{maximum_pause_duration}'] = dataset

    def set_dataset(self, mode:str, **kwargs):
        if (mode not in self.datasets) and (mode != 'long'):
            raise ValueError('`mode` should be in ["base","long"]')
        if mode in self.datasets:
            print(f'Switching to {mode}')
            self.dataset = self.datasets[mode]
            self.mode = mode
        elif mode == "long":
            self._create_long_dataset(**kwargs)
            mode = f"long_{kwargs.get('maximum_pause_duration',3) }"
            print(f'Switching to concatenated IPUs - {mode}')
            self.dataset = self.datasets[mode]
            self.mode = mode

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        start, stop, channel, audio = self.dataset[item]
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio, device=self.device)
        mel = get_mel(audio, device=self.device)
        return start, stop, channel, mel


# %%
def predict_extract(model, audio, s_start:float=0., s_stop:float=300., channel:int=0, 
            device=DEVICE, model_options=whisper.DecodingOptions()):
    fs = 16000
    audio = audio[channel, int(s_start*fs):int(s_stop*fs)]
    mel = get_mel(audio, device)
    results = model.decode(mel, model_options)
    return results



#%% ------ Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="File/Folder to Transcribe")
    parser.add_argument("--model_size", '-m', choices=['base','tiny','medium','small','large'], default="medium", help="Which whisper model to use")
    parser.add_argument("--use_ipus", '-ipus', action="store_true", help="Whether to run model.transcribe() (default) or on ipus (results not as good)")
    parser.add_argument("--save_res_csv", '-csv', action="store_true", help="Whether to save model predictions as csv")
    parser.add_argument("--do_bas_alignment", '-a', action="store_true", help="Whether to do BAS alignment")
    parser.add_argument("--language", '-l', type=str, default='fra-FR', help="Language option for BAS.")
    
    args = parser.parse_args()
    if not os.path.exists(args.filepath):
        raise ValueError("File does not exist")
    elif not os.path.isdir(args.filepath):
        args.filepath = [args.filepath]
    else: # filter so that only audio files are selected
        args.filepath = [os.path.join(args.filepath, x) for x in sorted(os.listdir(args.filepath)) if os.path.splitext(x)[-1] == '.wav']

    return args

#%% ------ Main
if __name__ == '__main__':
    args = parse_arguments()

    for filepath in args.filepath:
        print(f"\n---- Transcribing file {filepath}")
        algo_start = datetime.now().timestamp()
        dataset = AudioFileDataset(filepath)
        model = whisper.load_model(args.model_size)
        options = whisper.DecodingOptions(language=args.language.lower()[:2], without_timestamps=False, fp16 = False)

        if args.use_ipus:
            # use long ipus - worse result with short
            dataset.set_dataset("long")
            loader = torch.utils.data.DataLoader(dataset, batch_size=16)
            md_transcr = {'start':[], 'stop':[], 'speaker':[], 'text': []}
            for start, stop, channel, mels in tqdm(loader):
                results = model.decode(mels, options)
                # /!\ batches
                for k,v in zip(['start','stop','speaker','text'],[start, stop, channel, [result.text for result in results]]):
                    if not isinstance(v,list):
                        v = v.tolist()
                    md_transcr[k].extend(v)
            md_transcr = pd.DataFrame(md_transcr)

        else:
            # Run using whisper.transcribe()
            md_transcr = []
            for speaker in range(dataset.audio.shape[0]):
                result = model.transcribe(dataset.audio[speaker,:], fp16=False)
                tmp = pd.DataFrame(result["segments"])
                tmp['speaker'] = speaker
                md_transcr.append(tmp)
            md_transcr = pd.concat(md_transcr, axis=0).sort_values('start').reset_index(drop=True).rename(columns={'end':'stop'})

        print(f"Transcription duration: {np.round(algo_start - datetime.now().timestamp(),3)}s")

        if args.save_res_csv:
            fn = os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}-{'transcr' if not args.use_ipus else 'ipus'}-{args.model_size}.csv")
            md_transcr.to_csv(fn, index=False)

        audios = {speaker: os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{speaker}.wav") for speaker in range(dataset.nb_channels)}
        textgrids = {speaker: os.path.join(dataset.save_folder, f"{dataset.filename[:-4]}_mono_{speaker}.TextGrid") for speaker in range(dataset.nb_channels)}
        # save as TextGrid
        for speaker in range(dataset.nb_channels):
            # saving in different files to test alignment
            df = md_transcr[md_transcr.speaker == speaker]
            df.start = df.start.apply(lambda x: np.round(x,2))
            df.stop = df.stop.apply(lambda x: np.round(x,2)) # issues if no rounding
            overlaps = ((df['start'] - df['stop'].shift()).dropna() < 0)
            print('nb overlaps:', overlaps.sum())
            write_tier(df, file_name=textgrids[speaker], 
                    annot_tier=f'spk{speaker}', text_column='text', timestart_col='start', timestop_col='stop',
                    file_duration=dataset.audio_duration)

        if args.do_bas_alignment:
            for speaker in range(dataset.nb_channels):
                ns_path = audios[speaker]
                ts_path = textgrids[speaker]
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

        print(f"Total duration for file: {np.round(algo_start - datetime.now().timestamp(),3)}s")
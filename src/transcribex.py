import torch
import pandas as pd
import whisperx
import torchaudio
from datetime import datetime
import audiofile
import os
import numpy as np
import logging
import librosa

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)


afolder = 'brainkt-expe/data/audio'
afiles = os.path.listdir(afolder)
afiles = [os.path.join(afolder, x) if isinstance(x,str) else (os.path.join(afolder, x[0]),x[1]) for x in afiles]

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
#device='cpu'
logger.warning(f"device = {device}")
# transcribe with original whisper
model = whisperx.load_model("medium", device)

for afile in afiles:
    if isinstance(afile, str):
        channels = [0,1]
    else:
        afile, ch = afile
        channels = [ch]
    # read file
    sig, fs = librosa.load(afile, sr=16000, mono=False)
    #sig, fs = audiofile.read(afile, always_2d=True)
    for i in channels:
        #try:
            start = datetime.now()
            logger.warning(f'Starting file {afile} {i} at {start}')
            audio_file = sig[0,:]
            start = start.timestamp()
            result = model.transcribe(sig[i,:], language='fr')
            logger.warning(f'Transcription done.')
            # csv - transcribe now to skip errors
            t0 = pd.DataFrame(result["segments"])
            df_path = afile.replace('.wav',f'_{i}.csv')
            t0.to_csv(df_path, index=False)
            # load alignment model and metadata
            try:
                model_a, metadata = whisperx.load_align_model(language_code='fr', device=device) # result["language"]
                # align whisper output
                result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)
                logger.warning('Alignment done.')
                # csv
                t1 = pd.DataFrame(result_aligned["word_segments"])
                del result_aligned
                df_path = afile.replace('.wav',f'_{i}-word.csv')
                t1.to_csv(df_path, index=False)
            except Exception as e:
                logger.warning(f"\n---- Alignment errors file {afile}")
            print(f"File {afile} ch {i} - Duration: ", np.round(datetime.now().timestamp() - start, 3), "s")
            logger.warning(f"File {afile} ch {i} - Duration: {np.round(datetime.now().timestamp() - start, 3)}s")
            del result
        #except Exception as e:
        #    print(f"File {afile} ch {i} - Skip", e)

#!/usr/bin/python
"""
@author: Eliot Maës
@creation: 2022/11/11

Utility script. Contains functions to query BAS plateform and parse result
---------
**Using BAS**

From the [BAS Documentation Page](https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/help), several pipelines are available (with -- those that interest us the most):

```
-- runPipelineWithASR       runCOALA                runGetVersion
runCOALAGetTemplates        -- runASR               -- runPipeline
runTTS                      runSubtitle             runTextEnhance
runDoReCo                   runSpeakDiar            runVoiceActivityDetection
runChunkPreparation         runMAUSGetInventar      runASRGetQuota
runMINNI                    runMAUS                 getLoadIndicatorXML
runTextAlign                runMAUSBasic            -- runPho2Syl
getLoadIndicator            runAnnotConv            runFormantAnalysis
runChunker                  runMAUSGetHelp          runChannelSeparator
runTTSFile                  runAudioEnhance         runEMUMagic
runG2P                      runGetVersion           runAnonymizer
```

Calling those pipelines can be done using CURL:
```bash
curl -v -X POST -H 'content-type: multipart/form-data' -F com=yes -F INSKANTEXTGRID=true -F USETEXTENHANCE=true -F TARGETRATE=100000 -F TEXT=@[XXX.TextGrid] -F NOISE=0 -F PIPE=G2P_CHUNKER_MAUS_PHO2SYL -F aligner=hirschberg -F NOISEPROFILE=0 -F speakNumber=0 -F ASIGNAL=brownNoise -F NORM=true -F mauschunking=false -F INSORTTEXTGRID=true -F WEIGHT=default -F minanchorlength=3 -F LANGUAGE=eng-US -F USEAUDIOENHANCE=true -F maxlength=0 -F KEEP=false -F preference=-2.97 -F nrm=no -F LOWF=0 -F WHITESPACE_REPLACEMENT=_ -F marker=punct -F USEREMAIL=[XXX] -F boost=true -F MINPAUSLEN=5 -F forcechunking=false -F NOINITIALFINALSILENCE=false -F minVoicedLength=200 -F InputTierName=[XXX] -F OUTFORMAT=TextGrid -F syl=no -F ENDWORD=999999 -F minSilenceLength=200 -F wsync=yes -F UTTERANCELEVEL=false -F featset=standard -F INSPROB=0.0 -F OUTSYMBOL=x-sampa -F minchunkduration=15 -F SIGNAL=@[XXX.wav] -F stress=no -F MODUS=default -F RELAXMINDUR=false -F RELAXMINDURTHREE=false -F STARTWORD=0 -F INSYMBOL=sampa -F PRESEG=false -F AWORD=ANONYMIZED -F USETRN=false -F MAUSSHIFT=default -F HIGHF=0 -F silenceonly=0 -F boost_minanchorlength=4 -F ADDSEGPROB=false 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runPipeline'
```
"""

import requests
import xml.etree.ElementTree as ET
import ffmpeg # ffmpeg vs ffmpeg-python: both are imported with 'ffmpeg' but not the same commands
import os
import json
import subprocess

from textgrid_utils import read_tier, write_tier

BAS_ACCEPTED_LG = ['cat', 'deu', 'eng', 'fin', 'hat', 'hun', 'ita', 'mlt', 'nld', 'nze', 'pol', 
    'aus-AU', 'afr-ZA', 'sqi-AL', 'arb', 'eus-ES', 'eus-FR', 'cat-ES', 'nld-NL-GN', 'nld-NL', 
    'nld-NL-OH', 'nld-NL-PR', 'eng-US', 'eng-AU', 'eng-GB', 'eng-GB-OH', 'eng-GB-OHFAST', 'eng-GB-LE', 
    'eng-SC', 'eng-NZ', 'ekk-EE', 'kat-GE', 'fin-FI', 'fra-FR', 'deu-DE', 'gsw-CH-BE', 'gsw-CH-BS', 
    'gsw-CH-GR', 'gsw-CH-SG', 'gsw-CH-ZH', 'gsw-CH', 'hat-HT', 'hun-HU', 'isl-IS', 'ita-IT', 'jpn-JP', 
    'gup-AU, ''sampa', 'ltz-LU', 'mlt-MT', 'nor-NO', 'fas-IR', 'pol-PL', 'ron-RO', 'rus-RU', 'slk-SK', 
    'spa-ES', 'swe-SE', 'tha-TH', 'guf-AU']

class AlignTranscription():
    def __init__(self, audio_path:str, transcription_path:str, transcription_tier:str, 
                    lg:str='fra-FR', bas_option_path:str='bas-pipeline-config.json') -> None:
        self.audio_path = audio_path
        self.transcription_path = transcription_path
        self.transcription_tier = transcription_tier
        tp = os.path.splitext(self.transcription_path) # splitting
        self.aligned_transcription = f"{tp[0]}-bas{tp[1]}"
        if lg not in BAS_ACCEPTED_LG:
            raise ValueError(f'Language {lg} not accepted by BAS, must be one of {BAS_ACCEPTED_LG}')
        else:
            self.language = lg
        self.bas_option_path = bas_option_path

    def _compress_audio(self) -> None:
        """Compress audio and replace self.audio_path
        """
        audio_path_init = self.audio_path
        self.audio_path = os.path.splitext(self.audio_path) # splitting
        self.audio_path = f"{self.audio_path[0]}-compressed{self.audio_path[1]}"
        try:
            input = ffmpeg.input(audio_path_init)
            ffmpeg.output(input.audio, self.audio_path,
                    # extra arguments as kwargs, such as        
                    #**{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                ).overwrite_output().run()
        except AttributeError: # module 'ffmpeg' has no attribute 'input' because wrong module is loaded
            print("ffmpeg is loaded instead of ffmpeg-python. uninstall ffmpeg and reinstall ffmpeg-python")
            ffmpeg_call = ["ffmpeg", "-i", audio_path_init, self.audio_path]
            # call
            subprocess.call(ffmpeg_call, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
        

    def _curl_pipeline(self, notification_email:str=None, **kwargs) -> None:
        """Query BAS Pipeline with object audio/transcription. If error, returns query response; 
        otherwise downloads result into TextGrid file.
        """
        ### open files and add to options
        with open(self.bas_option_path,'r') as f: 
            gen_config = json.load(f)

        # transform to dict and add additional parameters
        spe_config = {
            "InputTierName":self.transcription_tier,
            "LANGUAGE":self.language, 
        }
        if notification_email is not None:
            spe_config["USEREMAIL"] = notification_email
        config = dict(gen_config, **spe_config)

        files = {
            "SIGNAL": open(self.audio_path, "rb"),
            "TEXT": open(self.transcription_path, "rb"),
        }
        ### query BAS
        SOURCE_URL = "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runPipeline"
        HEADER = {}
        response = requests.post(SOURCE_URL, files=files, data=config, headers = HEADER)
        # will get an email response anw - 200 is code if finished, whether failed or succeeded
        ### parse and download file
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            if (root[0].tag == 'success') and (root[0].text == 'true'):
                file_url = root[1].text #['WebServiceResponseLink']['downloadLink']
                dl_file = requests.get(file_url)
                open(self.aligned_transcription, "wb").write(dl_file.content)
            else:
                print(response.text)
        else:
            print(response.text)

    def _bas_to_ipu(self, tiers_to_keep:list=[], min_pause_duration:float=0.3, **kwargs) -> None:
        """
        Tiers in file:
            "ORT-MAU", "KAN-MAU", "KAS-MAU", "MAU", "MAS", "TRN"
        TRN is the given transcription, rewritten with different time windows (20-30s)

        Creates IPUs using word tier (ORT-MAU) 
        """
        outfile = self.aligned_transcription.replace('bas','bas-ipus')
        # read file
        words = read_tier(self.aligned_transcription, tier_name="ORT-MAU")
        # create ipus - columns start, stop
        words['pause_duration'] = (words['start'] - words['stop'].shift()).fillna(0.) > min_pause_duration
        words['ipu_id'] = words['pause_duration'].cumsum()
        ipus = words.groupby('ipu_id').agg({
            'start': 'min', 'stop': 'max', 'text': lambda x: ' '.join(list(x))
        })
        # write ipus
        write_tier(ipus, outfile)
        # write other tiers to file
        for tiername in tiers_to_keep:
            pass

    def run_pipeline(self, compress:bool=False, **kwargs):
        if compress:
            print('Compressing...', end=" ")
            self._compress_audio()
        # query
        print("Querying BAS...", end=" ")
        self._curl_pipeline(**kwargs)
        # check if file exists
        if os.path.exists(self.aligned_transcription):
            print("Concatenating to IPUS...", end= " ")
            self._bas_to_ipu(**kwargs)
            print("Done.")
        else:
            print("BAS Alignment Failed.")
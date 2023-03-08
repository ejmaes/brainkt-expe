#!/usr/bin/python
"""
@author: Eliot Maës
@creation: 2022/11/11

Read / Write Textgrid utility functions.
Reading can be done with SPPAS / Textgrid library.
Writing is done with SPPAS.
"""
import textgrid
import typing
import pandas as pd
import os, sys
from tqdm import tqdm

# Removing most of SPPAS output
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# SPPAS
SPPAS_PATH = "/Users/eliot/Documents/tools/SPPAS"
sys.path.append(SPPAS_PATH)
# reading / writing textgrids
import sppas.src.anndata.aio.readwrite as spp
import sppas.src.anndata as sad


#%% ---------- READING ----------
def read_tier(tg_path:str, tier_name:typing.Union[list,str]=None) -> pd.DataFrame:
    """Read tier and return as DataFrame.
    If no tier is given, returns the first tier.
    DataFrame includes a column for the source tier.
    """
    tg = textgrid.TextGrid.fromFile(tg_path)
    # get tier names
    tg_tiers = {tg[i].name : i for i in range(len(tg))}
    if tier_name is None: 
        print(f"Reading first tier from file: {tg[0].name}")
        tier_name = [tg[0].name]
    elif isinstance(tier_name,str):
        tier_name = [tier_name]
    # else isinstance(tier_name,list)
    # read from tier
    dialogs = []
    for tname in tier_name:
        tn = tg_tiers[tname]
        tier = tg[tn]
        for t in tier:
            if t.mark not in ["#",""]: # check that
                dialogs.append({
                    'file': tg_path.split('/')[-1],
                    'tier': tname,
                    'start': t.minTime,
                    'stop': t.maxTime,
                    'text': t.mark
                })
    # return
    return pd.DataFrame(dialogs)

#%% ---------- WRITING -----------
def write_tier(df:pd.DataFrame, file_name:str, 
        example_file:str='/Users/eliot/Documents/projects/multimodal-grounding/multimodalgrounding/utils/example.TextGrid', 
        annot_tier:str='TRS', 
        text_col:str='text', timestart_col:str='start', timestop_col:str='stop',
        file_duration:float=None, overwrite:bool=True, **kwargs):
    """Write data from a dataframe into a TextGrid file in intervals
    """ 
    if os.path.exists(file_name):
        tg = textgrid.TextGrid.fromFile(file_name)
        tg_tiers = {tg[i].name : i for i in range(len(tg))} 
        if annot_tier in tg_tiers and overwrite:
            print('Tier exists. Overwriting.')
        elif (annot_tier in tg_tiers):
            print('Tier exists. Changing tier name.')
    # Check if there are no ipus overlapping themselves
    overlaps = ((df[timestart_col] - df[timestop_col].shift()).fillna(1.) < 0)
    if overlaps.sum() > 0:
        print(df[overlaps][[timestart_col, timestop_col, text_col]])
        for idx, _ in df[overlaps].iterrows():
            print(df.loc[idx-2:idx+2,[timestart_col, timestop_col, text_col]])
        raise IndexError("Overlaps between several speakers exist in this DataFrame.")
    # Add silence rows in DataFrame
    stops = df[timestart_col].iloc[1:].tolist()
    starts = df[timestop_col].iloc[:-1].tolist()
    if df[timestart_col].iloc[0] > 0:
        stops = [df[timestart_col].iloc[0]] + stops
        starts = [0.0] + starts 
    if (file_duration is not None) and (file_duration > stops[-1]):
        stops.append(file_duration)
        starts.append(df[timestop_col].iloc[-1])
        print(df[timestop_col].iloc[-1])
    df_sil = pd.DataFrame({timestart_col: starts, timestop_col: stops})
    df_sil[text_col] = "#"
    df_sil = df_sil[df_sil[timestart_col] != df_sil[timestop_col]] # don't add void cells
    df = pd.concat([df, df_sil], axis=0).sort_values(by=[timestart_col]).reset_index(drop=True)
    # Create / Read file
    if os.path.exists(file_name):
        parser = spp.sppasTrsRW(file_name)
        tier_list = parser.read()
    elif not os.path.exists(example_file):
        raise ValueError(f'`example_file` argument must point to an existing file.')
    else:
        parser = spp.sppasTrsRW(example_file)
        tier_list = parser.create_trs_from_extension(example_file)
    # Create annotations
    tier = tier_list.create_tier(annot_tier)
    # Sequentially add rows
    for _,row in tqdm(df.iterrows()):
        if row[timestart_col] < row[timestop_col]: # rows throwing errors
            # Add row
            interval = sad.sppasInterval(sad.sppasPoint(row[timestart_col],0.0), sad.sppasPoint(row[timestop_col], 0.0))
            tier.create_annotation(sad.sppasLocation(interval), sad.sppasLabel(sad.sppasTag(row[text_col])))
        else:
            print("\nError printing the following row:")
            print(row)

    parser.set_filename(file_name)
    parser.write(tier_list)


def write_tiers(df:pd.DataFrame, file_name:str, tier_col:str, tier_values:list=None, **kwargs) ->None:
    """Loop over tiers to write in the same file
    """
    tier_values = tier_values if tier_values is not None else df[tier_col].unique()
    for tier_name in tier_values:
        write_tier(df[df[tier_col] == tier_name], file_name, annot_tier=tier_name, **kwargs)


def merge_files(d:dict, outfile_path:str) -> None:
    """Take tier(s) from separate files (eventually rename them) and write in a new file

    Input
    --------
    d: dict, with 
        * key: str - TextGrid file
        * value: Union[str,tuple,list] - tiers to copy. if tuple, then the second value is the new name
    
    outfile_path: str
    """
    # uniform
    for k,v in d.items():
        if isinstance(v,str):
            d[k] = (v,v) 
        elif isinstance(v,tuple):
            continue
        else: 
            d[k] = [vitem if isinstance(vitem,tuple) else (vitem, vitem) for vitem in v]
    # creating the new file
    for filename in d:
        for (old_tiername, new_tiername) in d[filename]:
            # TODO
            pass
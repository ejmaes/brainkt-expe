import pandas as pd
import os
from datetime import datetime
import cv2
import glob

VIDEO_FOLDERS = "../data/video"
TASKS = {
    "kb 1": "Clavier 1", 
    "kb 2": "Clavier 2", 
    "fils 1": "Fils 1", 
    "fils 2": "Fils 2", 
    "maze": "Labyrinthe", 
    "pwd": "Mot de passe", 
    "simon": "Simon"
}

def get_frame_rate(folder_path:str):
    # take from original videos
    fullpath = os.path.join(folder_path, '*/*.MXF')
    vids = list(glob.glob(fullpath))
    assert len(vids) > 0
    cap = cv2.VideoCapture(vids[0])
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    return framespersecond
    

def parse_video_markers(folder_path:str, framespersecond:int=25) -> pd.DataFrame:
    """Read markers, select adequate columns and compute diff from start marker
    """
    markers = pd.read_csv(os.path.join(folder_path, 'markers2.csv')) # TODO: remove
    markers = markers[['#', 'Source In', 'Color','Notes']]
    # note: the last number is in frames, out of frame rate. needs to be updated
    markers['Source In'] = markers['Source In'].apply(lambda x: f"{x[:8]}:{str(int(x[-2:])/framespersecond)[2:].ljust(3,'0')}")
    try:
        markers['dt'] = markers['Source In'].apply(lambda x: datetime.strptime(x, '%H:%M:%S:%f'))
    except ValueError as e:
        if "does not match format '%H:%M:%S:" in str(e): # partial error
            # time data '24:XX:YY:ZZZ' does not match format '%H:%M:%S:%f'
            # dt / Source In are not returned so modification can be applied
            # no need to worry about 0fill since 23-24
            markers['Source In'] = markers['Source In'].apply(lambda x: f"{int(x[:2])-1}{x[2:]}") 
            markers['dt'] = markers['Source In'].apply(lambda x: datetime.strptime(x, '%H:%M:%S:%f'))
        else: 
            raise e
    except IndexError as e:
        if "single positional indexer is out-of-bounds" in str(e):
            raise TypeError("Edit index was not correctly exported.")
        else: 
            raise e

    # Diff from video start
    markers['dt_from_0'] = markers.dt - markers.dt.iloc[0]
    markers['ts'] = markers['dt_from_0'].apply(lambda x: str(x).split()[-1])
    # to be used with EEG / Empatica for synchronisation
    markers['seconds_from_video_start'] = markers.dt_from_0.apply(lambda x: x.total_seconds())
    # in camera format
    markers['ts'] = markers['ts'].apply(lambda x: f"{x}:00" if len(x.split('.')) == 1 else ':'.join(x.split('.'))[:-4])
    
    # Diff from experiment start
    markers['dt_from_T1S'] = markers.dt - markers[markers.Notes == "Start Task 1"].dt.iloc[0]
    markers['seconds_from_expe_start'] = markers.dt_from_T1S.apply(lambda x: x.total_seconds())
    
    # Diff relative
    markers['task_duration'] = markers.seconds_from_expe_start - markers.seconds_from_expe_start.shift() 
    # update for Start / End Task 1 which aren't together anymore
    markers.loc[markers.Notes == 'End Task 1', 
                'task_duration'] = markers.loc[markers.Notes == 'End Task 1', 'seconds_from_expe_start'].iloc[0] - markers.loc[markers.Notes == 'Start Task 1', 'seconds_from_expe_start'].iloc[0]

    # Eventually rename / remove some markers - few files, will be done manually
    accepted_notes = ["Start","Watch p1","Watch p2","Clap","Start Task 1","End Task 1","End Task 2","Stop"]
    accepted_tasks = list(TASKS.keys())
    # tasks
    markers.Notes.apply(lambda x: print(f"{folder_path}: marker {x} needs to be checked") if x not in accepted_notes+accepted_tasks else None)

    # return
    return markers[['#','Color','Notes','ts','seconds_from_video_start','seconds_from_expe_start', 'task_duration']]


#%% ------ Audio
import subprocess
import audiofile

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
    subprocess.call(ffmpeg_call, 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    # return name
    return f"{filename}.{output_ext}"

#%% ------ Main
extract_audio = False
if __name__ == '__main__':
    # parse video folder for 'markers.csv' files, parse them, and add to metadata file
    dyads = sorted([x for x in os.listdir(VIDEO_FOLDERS) if os.path.isdir(os.path.join(VIDEO_FOLDERS, x))])
    print("Existing dyads:", dyads)
    markers = []
    for file in dyads:
        # MARKERS
        markers_path = os.path.join(VIDEO_FOLDERS, file)
        try:
            df = parse_video_markers(markers_path) #, get_frame_rate(markers_path)
            df['file'] = file
            markers.append(df)
        except FileNotFoundError as e:
            print(f"Cannot read marker file from {markers_path}")
            print(type(e), e)

        # AlSO EXTRACT AUDIO
        if extract_audio:
            [date, group] = file.split('_')
            video = os.path.join(VIDEO_FOLDERS, file, f"bkt-{date}-{group}.mov")
            vaudio_name = convert_video_to_audio_ffmpeg(video)
            vaudio_name = convert_video_to_audio_ffmpeg(video, audio_stream_to_map=2)
    
    markers = pd.concat(markers, axis=0).reset_index(drop=True)
    # tasks
    tasks = markers[markers.Notes.isin(TASKS.keys())]
    tasks.Notes = tasks.Notes.apply(lambda x: TASKS[x])
    ptasks = tasks.groupby(['file','Notes']).agg({'task_duration': 'sum', 'Color': lambda x: (len(list(x)) > 0) and (list(x)[-1] == 'Mint') }).reset_index(
        drop=False).rename(columns={'Color':'completed'})
    ptasks = ptasks.pivot(index="file", columns="Notes", values=["completed", "task_duration"])
    ptasks = ptasks.fillna(value={col:False for col in ptasks.columns if col[0] == 'completed'}).swaplevel(axis=1)
    ptasks = ptasks.reindex(sorted(ptasks.columns), axis=1)
    # pivot with ts from video start
    columns_order = [x for x in markers.Notes.unique().tolist() if x not in TASKS.keys()] # in order
    markers_from_start = markers[markers.Notes.isin(columns_order)].pivot(index="file", columns="Notes",values="seconds_from_video_start")[columns_order]
    # pivot with ts from expe start
    markers_from_estart = markers[(markers.Notes.isin(columns_order)) & (markers.seconds_from_expe_start >= 0)]
    columns_order = [x for x in markers_from_estart.Notes.unique().tolist() if x not in TASKS.keys()] # in order
    markers_from_estart = markers_from_estart.pivot(index="file", columns="Notes",values="seconds_from_expe_start")[columns_order]
    # Export: 
    markers.to_csv(os.path.join(VIDEO_FOLDERS, 'all_markers.csv'), index=False)
    tasks.reset_index(drop=True).to_csv(os.path.join(VIDEO_FOLDERS, 'all_markers_tasks.csv'), index=False)
    ptasks.to_excel(os.path.join(VIDEO_FOLDERS, 'tasks_duration_completion.xlsx'))
    markers_from_start.reset_index(drop=False).to_csv(os.path.join(VIDEO_FOLDERS, 'markers_from_video_start.csv'), index=False)
    markers_from_estart.reset_index(drop=False).to_csv(os.path.join(VIDEO_FOLDERS, 'markers_from_experiment_start.csv'), index=False)

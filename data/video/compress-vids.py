import os
import glob
import ffmpeg
import subprocess
import re
import time
import numpy as np

#PAT = re.compile("time=[0-9:\.]{8}")

def sub_execute(int_path, out_path):
    start = time.time()
    p = subprocess.run([
            'ffmpeg', '-i', 
            int_path,
            '-c:v', 'libx265', '-preset', 'fast', '-crf', '28', '-tag:v', 'hvc1', '-c:a', 'eac3', '-b:a', '224k',
            out_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    #x = re.findall(PAT, p.__str__())
    t = time.time() - start
    if "error" in p.__str__():
        print(p.__str__())
    print(f'Took {int(t//60)}min{np.round(t%60,2)}s; finished at {time.strftime("%H:%M:%S",time.localtime())}\n')


or_path = os.path.expanduser("~/Documents/projects/brainkt-expe/data/video/")
targ_path = os.path.expanduser("~/Documents/projects/brainkt-expe/data/video-compressed/")
for folder in sorted(os.listdir(or_path)):
    files = glob.glob(os.path.join(or_path, folder, '*.mov'))
    if (len([x for x in files if '-rev' in x]) > 0) and folder not in os.listdir(targ_path):
        targ_folder = os.path.join(targ_path, folder)
        os.makedirs(targ_folder)
        date, group = folder.split('_') 
        for pat in ['rev', group[0:2], group[2:4]]: # doing the other ones some other day
            in_path = os.path.join(or_path, folder, f"bkt-{date}-{group}-{pat}.mov")
            out_path = os.path.join(targ_folder, f"bkt-{date}-{group}-{pat}.mp4")
            if pat == 'rev':
                out_path = out_path.replace('-rev','')
            print(in_path, '>>>', out_path)
            sub_execute(in_path, out_path)

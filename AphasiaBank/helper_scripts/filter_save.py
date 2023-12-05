'''
Filter save files
Keep most recent save file
'''
import os
from datetime import datetime
import shutil
from tqdm import tqdm


def ckpt_filter(save_dir):
    ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith('CKPT')]
    sorted_subdirs = sorted(ckpt_dirs, key=lambda x: datetime.strptime(x.split('+')[1], '%Y-%m-%d'))


    for i in range(len(sorted_subdirs)-1):
        del_dir = f"{save_dir}/{sorted_subdirs[i]}"
        shutil.rmtree(del_dir)


def filter_AB():
    EXP_DIR="/home/mkperez/speechbrain/AphasiaBank/results/Frid"

    for sdir in tqdm([d for d in os.listdir(EXP_DIR) if d != 'base']):
        for fold_dir in [d for d in os.listdir(f"{EXP_DIR}/{sdir}") if d.startswith('Fold')]:
            ckpt_filter(f"{EXP_DIR}/{sdir}/{fold_dir}/save")

if __name__=="__main__":
    filter_AB()

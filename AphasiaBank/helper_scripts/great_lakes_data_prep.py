'''
Update data paths for great lakes
'''

import pandas as pd
from tqdm import tqdm
import os

DATA_DIR="/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

def replace_csv(src_csv):
    # overwrite wav path
    df = pd.read_csv(src_csv)
    df['wav'] = [w.replace('/z/mkperez/speechbrain/','/home/mkperez/speechbrain/') for w in df['wav'].tolist()]

    df['exists'] = df['wav'].apply(os.path.exists)
    df = df[df['exists']].drop(columns='exists')

    df.to_csv(src_csv)


def data_paths_Frid():
    for fold in range(1,13):
        for stype in ['train', 'dev', 'test']:
            # for ptype in ['pn', 'p', 'n', 'multi']:
            for ptype in ['multi']:
                src_csv = f"{DATA_DIR}/Fold_{fold}/{stype}_{ptype}.csv"

                replace_csv(src_csv)

PROTO_DATA_DIR="/home/mkperez/speechbrain/AphasiaBank/data/Proto-clinical"
def data_paths_Proto():
    for stype in ['train', 'dev', 'test']:
        src_csv = f"{PROTO_DATA_DIR}/{stype}.csv"
        # replace_csv(src_csv)


        # mtl_paraphasia
        for ptype in ['multi','n','p','pn','s']:
            mtl_paraphasia_csv = f"{PROTO_DATA_DIR}/mtl_paraphasia/{stype}_{ptype}.csv"
            replace_csv(mtl_paraphasia_csv)

if __name__ == "__main__":
    data_paths_Proto()
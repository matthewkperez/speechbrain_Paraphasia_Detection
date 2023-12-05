'''
Prepare colab files:
    -segment test files and store in separate dir

'''

import os
import pandas as pd
import shutil
from tqdm import tqdm


test_data_dir = "/z/mkperez/speechbrain/AphasiaBank/data/no_paraphasia/test.csv"
df = pd.read_csv(test_data_dir)
print(df.columns)

wdir = "/z/mkperez/speechbrain/AphasiaBank/data/no_paraphasia/colab_dir"
os.makedirs(wdir, exist_ok=True)

for i,row in tqdm(df.iterrows(), total=len(df)):
    # print(row['wav'])
    # exit()
    og_wav = row['wav']
    suffix_wav = og_wav.split("/")[-1]
    shutil.copyfile(row['wav'],f"{wdir}/{suffix_wav}")

    # if i ==5:
    #     exit()

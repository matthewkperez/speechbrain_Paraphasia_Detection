'''
Given list of wer.txt outputs from SB
Generate WER for each severity class (mild, mod, sev, v. sev.)
'''

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from hyperpyyaml import load_hyperpyyaml
import jiwer

import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter

def dict_read(filepath):
    d = {}
    with open(filepath, 'r') as r:
        for line in r.readlines():
            d[line.split()[0]] = line.split()[1]
    return d

def generate_wer_summary(filepath,spk2subtype,spk2sev,spk2aq):
    '''
    Output dataframe with severity, and wer for each utt
    filepath = speechbrain wer.txt filepath
    '''

    df_lst = []
    with open(filepath, 'r') as r:
        for line in tqdm(r.readlines()):
            line = line.strip()
            if len(line.split()) == 14 and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                wer = float(line.split()[2])
                spkr_id = utt_id.split("-")[0]
                err = int(line.split()[4])
                tot = int(line.split()[6][:-1])


                # need both severity and subtype to add
                if spkr_id in spk2sev:
                    # subtype = spk2subtype[spkr_id]
                    # if len(subtype) == 2:
                    #     subtype = 'not aphasic'

                    
                    # aq = float(spk2aq[spkr_id])
                    sev = spk2sev[spkr_id]




                    df_loc = pd.DataFrame({
                        'utt_id': [utt_id],
                        'spkr': [spkr_id],
                        'wer': [wer],
                        'sev': [sev],
                        'err': [err],
                        'tot': [tot]
                    })
                    df_lst.append(df_loc)

    df = pd.concat(df_lst)
    return df

def analyze_SB_wer():
    EXP_DIR="/home/mkperez/speechbrain/AphasiaBank/results/Table-1_verbatim/ES_Transformer-Transformer"

    wer_file = f"{EXP_DIR}/wer.txt"
    outfile = f"{EXP_DIR}/wer_sev_stats.txt"
    
    # load const
    spk2aq = dict_read("/home/mkperez/speechbrain/AphasiaBank/spk2aq")
    spk2sev = dict_read("/home/mkperez/speechbrain/AphasiaBank/spk2sevbin")
    spk2subtype = dict_read("/home/mkperez/speechbrain/AphasiaBank/spk2subtype")

    df_exp = generate_wer_summary(wer_file,spk2subtype,spk2sev,spk2aq)
    group_df_err = df_exp.groupby('sev')['err'].sum().reset_index()
    group_df_tot = df_exp.groupby('sev')['tot'].sum().reset_index()

    # df_merged = pd.concat([group_df_err,group_df_tot])
    df_merged = pd.merge(group_df_err, group_df_tot, on='sev', how='inner')

    with open(outfile, 'w') as w:
        for i, row in df_merged.iterrows():

            print(f"{row['sev']}: {row['err']/row['tot']}")
            w.write(f"{row['sev']}: {row['err']/row['tot']}\n")



if __name__ == "__main__":
    analyze_SB_wer()
    # analyze_pykaldi_wer()
    # analyze_whisperX_wer()
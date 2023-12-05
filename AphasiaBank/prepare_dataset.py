'''
Prepare dataset files (csvs)

entries include:
utt_ID, duration, wav_path, speaker_ID, transcription
'''
from lib2to3.pgen2.pgen import DFAState
import os
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import math
import chaipy.io
import shutil
from tqdm import tqdm
import re
import subprocess
import csv


def edit_csvs(root_dir):
    ## Add info to current csvs

    for stitle in ['train', 'val', 'test']:
        og_file = f"{root_dir}/{stitle}_strat.csv"
        new_file = f"{root_dir}/{stitle}_strat_sevnum.csv"
        df = pd.read_csv(og_file)

        sev_dict = {'control':0, 'mild':1, 'moderate':2, 'severe':3, 'v_severe':4, 'unk':-1}
        df['severity_cat'] = ['control' if row['dataset'] == 'Control' else row['severity_cat'] for i, row in df.iterrows()]
        df['severity_cat_num'] = [int(sev_dict[row['severity_cat']]) for i, row in df.iterrows()]
        df['wrd'] = [row['wrd'].strip() for i, row in df.iterrows()]
        df.to_csv(new_file)


def write_text_file(text_list,outfile):
    with open(outfile,'w') as w:
        for text in text_list:
            w.write(text)
    
def make_dicts(sb_data_dir):
    scores_xlsx_path = "/z/public/data/AphasiaBank/spkr_info/updated_scores.xlsx"
    df_scores = pd.read_excel(scores_xlsx_path, sheet_name='Time 1')
    df_scores_r = pd.read_excel(scores_xlsx_path, sheet_name='Repeats')

    spk2aq_str = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2aq_str2 = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores_r.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}

    spk2type_str = {row['Participant ID']:row['WAB Type'] for i, row in df_scores.iterrows() if row['WAB Type'] != 'U'}
    spk2type_str2 = {row['Participant ID']:row['WAB Type'] for i, row in df_scores_r.iterrows() if row['WAB Type'] != 'U'}

    spk2aq_str.update(spk2aq_str2)
    spk2type_str.update(spk2type_str2)

    chaipy.io.dict_write(fname=f"spk2sevbin",key2val=spkr2sev)
    chaipy.io.dict_write(fname=f"spk2subtype",key2val=spk2type_str)



    # og_metadata = os.path.join(OG_data_dir, "metadata.csv")
    # sb_data_dir = "/y/mkperez/speechbrain/AphasiaBank/data/speechbrain"
    # if not os.path.exists(sb_data_dir):
    #     os.makedirs(sb_data_dir)

    # df = pd.read_csv(og_metadata,header=0, skipfooter=0)
    # df = df[df['file_name'].str.endswith('.wav')] 
    # df = df.reset_index(drop=True)
    # print(df)
    # exit()
    spkr2sev = {}
    spkr2subtype = {}
    sev_count = {}
    subtype_count = {}
    for i,row in tqdm(df.iterrows(), total=len(df)):
        # severity
        if row['speaker'] in spk2aq_str:
            sev_cat = pd.cut([float(spk2aq_str[row['speaker']])], bins=[0,25,50,75,100], labels=["v_severe", "severe", "moderate", "mild"])[0]
            spkr2sev[row['speaker']] = sev_cat
            
            
            if sev_cat not in sev_count:
                sev_count[sev_cat] = 0
            sev_count[sev_cat]+=1
        elif row['dataset'] == 'Control':
            spkr2sev[row['speaker']] = 'control'
        else:
            spkr2sev[row['speaker']] = 'unk'

        # type
        if row['speaker'] in spk2type_str:
            subtype = spk2type_str[row['speaker']].strip()
            spkr2subtype[row['speaker']] = subtype
            
            
            if subtype not in subtype_count:
                subtype_count[subtype] = 0
            subtype_count[subtype]+=1
        elif row['dataset'] == 'Control':
            spkr2subtype[row['speaker']] = 'control'
        else:
            spkr2subtype[row['speaker']] = 'unk'

    # write dicts
    chaipy.io.dict_write(fname=f"spk2sevbin",key2val=spkr2sev)
    chaipy.io.dict_write(fname=f"spk2subtype",key2val=spk2type_str)

    # count
    df_sev_list = []
    tot_sum = sum([v for k,v in sev_count.items()])
    for k,v in sev_count.items():
        df_loc = pd.DataFrame({
            'sev': [k],
            'count':[v],
            'pct':[v/tot_sum]
            # k: [v,v/tot_sum]
        })
        df_sev_list.append(df_loc)
    df_sev = pd.concat(df_sev_list)
    df_sev = df_sev.set_index('sev').sort_index()
    print(df_sev)
    print(df_sev.transpose().round(2))

    # count subtype
    df_sev_list = []
    tot_sum = sum([v for k,v in subtype_count.items()])
    for k,v in subtype_count.items():
        df_loc = pd.DataFrame({
            'sev': [k],
            'count':[v],
            'pct':[v/tot_sum]
            # k: [v,v/tot_sum]
        })
        df_sev_list.append(df_loc)
    df_sev = pd.concat(df_sev_list)
    df_sev = df_sev.set_index('sev').sort_index()
    print(df_sev)
    print(df_sev.transpose().round(2))


def process_data(OG_data_dir):
    scores_xlsx_path = "/z/public/data/AphasiaBank/spkr_info/updated_scores.xlsx"
    df_scores = pd.read_excel(scores_xlsx_path, sheet_name='Time 1')
    df_scores_r = pd.read_excel(scores_xlsx_path, sheet_name='Repeats')
    spk2aq_str = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2aq_str2 = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores_r.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2type_str = {row['Participant ID']:row['WAB Type'] for i, row in df_scores.iterrows() if row['WAB Type'] != 'U'}
    spk2type_str2 = {row['Participant ID']:row['WAB Type'] for i, row in df_scores_r.iterrows() if row['WAB Type'] != 'U'}
    spk2aq_str.update(spk2aq_str2)
    spk2type_str.update(spk2type_str2)

    og_metadata = os.path.join(OG_data_dir, "metadata.csv")

    df = pd.read_csv(og_metadata,header=0, skipfooter=0)
    df = df[df['file_name'].str.endswith('.wav')] 
    df = df.reset_index(drop=True)


    utt_ids = []
    wavs = []
    durations = []
    severity = []
    severity_type = []
    unk_speakers = []
    severity_cat = []
    for i,row in tqdm(df.iterrows(), total=len(df)):

        rel_wav = row['file_name']
        utt_ids.append(row['file_name'].split("/")[-1].split(".")[0])
        absolute_filepath = os.path.join(OG_data_dir, rel_wav)
        wavs.append(absolute_filepath)

        # print(row['transcription'])
        # row['transcription'] = row['transcription'].replace("<brth>", "!")
        # row['transcription'] = row['transcription'].replace("<flr>", "@")
        # row['transcription'] = row['transcription'].replace("<lau>", "#")
        # row['transcription'] = row['transcription'].replace("<spn>", "$")
        row['transcription'] = row['transcription'].replace("<brth>", "")
        row['transcription'] = row['transcription'].replace("<flr>", "")
        row['transcription'] = row['transcription'].replace("<lau>", "")
        row['transcription'] = row['transcription'].replace("<spn>", "")

        # row['transcription'] = row['transcription'].replace("<brth>", "")
        # row['transcription'] = row['transcription'].replace("<flr>", "")
        # row['transcription'] = row['transcription'].replace("<lau>", "")
        # row['transcription'] = row['transcription'].replace("<spn>", "")

        # exit()

        # duration
        dur = librosa.get_duration(filename=absolute_filepath)
        durations.append(dur)

        # severity
        if row['speaker'] in spk2aq_str:
            severity.append(float(spk2aq_str[row['speaker']]))
            sev_cat = pd.cut([float(spk2aq_str[row['speaker']])], bins=[0,25,50,75,100], labels=["v_severe", "severe", "moderate", "mild"])
            severity_cat+=sev_cat
        else:
            severity.append("unk")
            severity_cat.append("unk")
            if row['dataset'] == 'Aphasia':
                unk_speakers.append(row['speaker'])

        # type
        if row['speaker'] in spk2type_str:
            severity_type.append(spk2type_str[row['speaker']])
        else:
            severity_type.append("unk")
        
            if row['dataset'] == 'Aphasia':
                unk_speakers.append(row['speaker'])

    df['ID'] = utt_ids
    df['wav'] = wavs
    df['duration'] = durations
    df['severity'] = severity
    df['aphasia_type'] = severity_type
    df['severity_cat'] = severity_cat
    df['group'] = [re.split(r'(\d+)', x.split("-")[0])[0] for x in df['ID']]
    df = df.rename(columns={'transcription':'wrd', 'speaker':'spk_id'})
    df = df.drop(columns=['file_name'])

    # drop empty utts
    df = df[~df['wrd'].str.isspace()]

    # severity cat
    sev_dict = {'control':0, 'mild':1, 'moderate':2, 'severe':3, 'v_severe':4, 'unk':-1}
    df['severity_cat'] = ['control' if row['dataset'] == 'Control' else row['severity_cat'] for i, row in df.iterrows()]
    df['severity_cat_num'] = [int(sev_dict[row['severity_cat']]) for i, row in df.iterrows()]
    df['wrd'] = [row['wrd'].strip() for i, row in df.iterrows()]

    return df

        
def personalization_main(sb_data_dir, OG_data_dir):
    print("Personalization")

    if not os.path.exists(sb_data_dir):
        os.makedirs(sb_data_dir)

    seed = 5
    df = process_data(OG_data_dir)

    # print(f"groups: {set(df['group'].values)}")
    df_no_kansas = df[df['group'] != 'kansas']
    df_kansas = df[df['group'] == 'kansas']
    # print(df)
    notest_kansas, test_kansas = train_test_split(df_kansas, test_size=0.2, random_state=seed, stratify=df_kansas['spk_id'])
    train_kansas, val_kansas = train_test_split(notest_kansas, test_size=0.2, random_state=seed, stratify=notest_kansas['spk_id'])
    train_no_kansas, val_no_kansas = train_test_split(df_no_kansas, test_size=0.2, random_state=seed, stratify=df_no_kansas['spk_id'])

    train_all = pd.concat([train_kansas, train_no_kansas])

    train_all.to_csv(f"{sb_data_dir}/train_all.csv", index=False)    
    train_kansas.to_csv(f"{sb_data_dir}/train_kansas.csv", index=False)
    val_kansas.to_csv(f"{sb_data_dir}/val_kansas.csv", index=False)
    test_kansas.to_csv(f"{sb_data_dir}/test_kansas.csv", index=False)

    train_no_kansas.to_csv(f"{sb_data_dir}/train_no_kansas.csv", index=False)
    val_no_kansas.to_csv(f"{sb_data_dir}/val_no_kansas.csv", index=False)


# Partition data and add subtpye and severity
def main(sb_data_dir,OG_data_dir):

    if not os.path.exists(sb_data_dir):
        os.makedirs(sb_data_dir)

    seed = 5
    df = process_data(OG_data_dir)



    # partition - > TODO: use control data for train only
    notest, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['severity_cat'])
    train, val = train_test_split(notest, test_size=0.05, random_state=seed, stratify=notest['severity_cat'])


    train.to_csv(f"{sb_data_dir}/train_strat.csv", index=False)
    val.to_csv(f"{sb_data_dir}/val_strat.csv", index=False)
    test.to_csv(f"{sb_data_dir}/test_strat.csv", index=False)

    # create json files for tokenizer
    train = train.rename(columns={'wrd':'words','duration':'length'})
    val = val.rename(columns={'wrd':'words','duration':'length'})
    test = test.rename(columns={'wrd':'words','duration':'length'})
    train = train.set_index('ID')
    val = val.set_index('ID')
    test = test.set_index('ID')
    train.to_json(f"{sb_data_dir}/train_strat.json", orient='index')
    val.to_json(f"{sb_data_dir}/val_strat.json", orient='index')
    test.to_json(f"{sb_data_dir}/test_strat.json", orient='index')

    # LM
    write_text_file(train['words'],f"{sb_data_dir}/train_strat.txt")
    write_text_file(val['words'],f"{sb_data_dir}/val_strat.txt")
    write_text_file(test['words'],f"{sb_data_dir}/test_strat.txt")



def segment_data(seg_file,target_dir,stype):
    seg_wav_dir = f"{target_dir}/wavs"
    if not os.path.exists(seg_wav_dir):
        os.makedirs(seg_wav_dir)

    # owav
    control_wav_scp="/z/mkperez/AphasiaBank/Duc_kaldi_prep/kaldi_data/Control/owav.scp"
    aph_wav_scp="/z/mkperez/AphasiaBank/Duc_kaldi_prep/kaldi_data/Aphasia/owav.scp"

    # create spkr->src wav dict
    spkr_src_wav_dict={}
    for scp in [control_wav_scp, aph_wav_scp]:
        with open(scp, 'r') as r:
            lines = r.readlines()
            for line in tqdm(lines, total=len(lines)):
                spkr_id = line.split()[0]
                wav_path = line.split()[1]
                spkr_src_wav_dict[spkr_id]=wav_path

    missed_files=[]
    id2wav = {}
    with open(seg_file, 'r') as r:
        # get control/aphasia and speaker
        lines = r.readlines()
        for line in tqdm(lines):
            utt_id = line.split()[0]
            speaker = line.split()[1]
            start_t = float(line.split()[2])
            end_t = float(line.split()[3])
            dur = end_t-start_t


            src_wav_path = spkr_src_wav_dict[speaker]

            new_seg_wav_path = f"{seg_wav_dir}/{utt_id}.wav"

            # sox segmentation
            cmd = ['sox', src_wav_path, new_seg_wav_path, 'trim', f"{start_t}", f"={end_t}"]
            list_files = subprocess.run(cmd)
            # assert list_files.returncode == 0, f"sox error {utt_id}: {start_s} = {end_s}"
            if list_files.returncode != 0:
                missed_files.append(utt_id)
                continue
            


            id2wav[utt_id] = (new_seg_wav_path,dur)

    # output missed files
    with open(f'missed_files_{stype}.txt', 'w') as w:
        for line in missed_files:
            w.write(f"line\n")

    return id2wav

def read_text_data(text_file):
    id2text={}
    with open(text_file, 'r') as r:
        lines = r.readlines()
        for line in tqdm(lines):
            id = line.split()[0]
            text = line.split(" ", 1)[1]

            print(line)
            print(id)
            print(text)
            id2text[id] = text
    return id2text


def prepare_duc_data():
    duc_src_dir="/y/mkperez/AphasiaBank/kaldi_data_updated/Aphasia+Control/ASR_Fold_5"
    target_dir="/z/mkperez/speechbrain/AphasiaBank/data/Duc_process"

    # scores
    scores_xlsx_path = "/z/public/data/AphasiaBank/spkr_info/updated_scores.xlsx"
    df_scores = pd.read_excel(scores_xlsx_path, sheet_name='Time 1')
    df_scores_r = pd.read_excel(scores_xlsx_path, sheet_name='Repeats')
    spk2aq_str = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2aq_str2 = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores_r.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2type_str = {row['Participant ID']:row['WAB Type'] for i, row in df_scores.iterrows() if row['WAB Type'] != 'U'}
    spk2type_str2 = {row['Participant ID']:row['WAB Type'] for i, row in df_scores_r.iterrows() if row['WAB Type'] != 'U'}

    spk2aq_str.update(spk2aq_str2)
    spk2type_str.update(spk2type_str2)


    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    

    for stype in ['dev','train', 'test']:
        csv_path = f"{target_dir}/{stype}.csv"
        csv_file = open(csv_path, 'w')
        csv_writer = csv.writer(csv_file)
        headerdata = ['wrd', 'dataset', 'spk_id', 'ID', 'wav', 'duration','severity','aphasia_type','severity_cat','group','speaker']
        csv_writer.writerow(headerdata)

        seg_file = f"{duc_src_dir}/{stype}/segments"
        text_file = f"{duc_src_dir}/{stype}/text"
        id2wav = segment_data(seg_file,target_dir,stype)
        id2text = read_text_data(text_file)


        for utt_id in id2wav.keys():
            group = re.split(r'(\d+)', utt_id)[0]
            speaker = utt_id.split("-")[0]
            print(utt_id)
            print(dataset)
            exit()
            # wrd,dataset,spk_id,ID,wav,duration,severity,aphasia_type,severity_cat,group,severity_cat_num
            csv_data = [id2text[utt_id],dataset,speaker,id2wav[utt_id][0],id2wav[utt_id][1]]
            label_data = [spk2aq_str[speaker],spk2type_str[speaker],severity_cat,group,severity_cat_num]


if __name__ == "__main__":
    # OG_data_dir = "/y/mkperez/AphasiaBank/data/seg_audio_HF"
    OG_data_dir = "/y/mkperez/AphasiaBank/data/seg_audio_torre"
    # OG_data_dir = "/y/mkperez/AphasiaBank/data/seg_audio_torre_clean"
    root_dir = "/y/mkperez/speechbrain/AphasiaBank/data/personalization_torre"
   
   
    # print(f"prepare: {root_dir}")
    # # main(root_dir, OG_data_dir)
    # personalization_main(root_dir, OG_data_dir)


    prepare_duc_data()




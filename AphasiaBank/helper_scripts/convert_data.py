'''
Prepare dataset files (csvs)
Convert standard kaldi dir to speechbrain dir

entries include:
utt_ID, duration, wav_path, speaker_ID, transcription
'''
from lib2to3.pgen2.pgen import DFAState
import os
import pandas as pd
import math
import chaipy.io
import shutil
from tqdm import tqdm
import re
import subprocess
import csv
from collections import Counter

def segment_data(root_dir,seg_file,target_dir,stype,seg_wav_bool):
    '''
    segments wavs and store new wav_path
    return id2wav which has {utt_id: (wav_path,dur,HC_AB)}
    '''
    seg_wav_dir = f"{target_dir}/wavs"
    if not os.path.exists(seg_wav_dir):
        os.makedirs(seg_wav_dir)

    # owav
    control_wav_scp=f"{root_dir}/Control/owav.scp"
    aph_wav_scp=f"{root_dir}/Aphasia/owav.scp"

    # create spkr->src wav dict
    spkr_src_wav_dict={}
    HC_AB_dict = {}
    for scp, AB_HC in zip([control_wav_scp, aph_wav_scp], ['Control', 'Aphasia']):
        with open(scp, 'r') as r:
            lines = r.readlines()
            for line in tqdm(lines, total=len(lines)):
                spkr_id = line.split()[0]
                wav_path = line.split()[1]
                spkr_src_wav_dict[spkr_id]=wav_path
                HC_AB_dict[spkr_id] = AB_HC


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


            if seg_wav_bool:
                # sox segmentation
                cmd = ['sox', src_wav_path, new_seg_wav_path, 'trim', f"{start_t}", f"={end_t}"]
                list_files = subprocess.run(cmd)
                # assert list_files.returncode == 0, f"sox error {utt_id}: {start_s} = {end_s}"
                if list_files.returncode != 0:
                    missed_files.append(utt_id)
                    continue
            


            id2wav[utt_id] = (new_seg_wav_path,dur,HC_AB_dict[speaker])

    # output missed files
    if len(missed_files) > 1:
        with open(f'missed_files_{stype}.txt', 'w') as w:
            for line in missed_files:
                w.write(f"line\n")

    return id2wav

def norm_text(text):
    text = text.lower().strip()


    return text

def read_text_data(text_file):
    '''
    output id2text = {utt_id: text}
    '''
    id2text={}
    with open(text_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            id = line.split()[0]
            text = line.split(" ", 1)[1]

            id2text[id] = norm_text(text)
            # print(f"pre: {text}")
            # print(f"norm: {id2text[id]}")
            # exit()

    return id2text

def check_names(text):
    if 'lastname' in text or 'firstname' in text:
        # print("name found")
        return False
    return True

def prepare_duc_data():
    root_dir = "/z/mkperez/AphasiaBank/kd_updated_para"
    duc_src_dir=f"{root_dir}/Aphasia+Control/ASR_fold_5_filtered_10s" # need segments, text, and owav.scp file
    target_dir="/z/mkperez/speechbrain/AphasiaBank/data/no_paraphasia"
    seg_wav_bool = False

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


    # make target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # make train, dev, test csvs
    for stype in ['dev','train', 'test']:
        csv_path = f"{target_dir}/{stype}.csv"
        csv_file = open(csv_path, 'w')
        csv_writer = csv.writer(csv_file)
        # headerdata = ['wrd', 'dataset', 'spk_id', 'ID', 'wav', 'duration','severity','aphasia_type','severity_cat','group','speaker']
        headerdata = ['wrd', 'HC_AB', 'spk_id', 'ID', 'wav', 'duration','severity','aphasia_type','group','severity_cat']
        csv_writer.writerow(headerdata)

        seg_file = f"{duc_src_dir}/{stype}/stored_segments"
        text_file = f"{duc_src_dir}/{stype}/text"
        id2wav = segment_data(root_dir,seg_file,target_dir,stype,seg_wav_bool)
        id2text = read_text_data(text_file)

        # contains both control and aph speakers
        for utt_id in id2wav.keys():
            group = re.split(r'(\d+)', utt_id)[0]
            speaker = utt_id.split("-")[0]


            if utt_id in id2text and check_names(id2text[utt_id]):
                # wrd,dataset,spk_id,ID,wav,duration,severity,aphasia_type,severity_cat,group,severity_cat_num
                meta_data = [id2text[utt_id],id2wav[utt_id][2],speaker,utt_id,id2wav[utt_id][0],round(id2wav[utt_id][1],2)]
                if id2wav[utt_id][2] == "Aphasia":
                    if speaker not in spk2aq_str:
                        aq = 'N/A'
                        sev_cat = -1
                    else:
                        aq = round(spk2aq_str[speaker],2)
                        sev_cat = pd.cut([spk2aq_str[speaker]], bins=[0,25,50,75,100], labels=[4,3,2,1])[0]
                    
                    if speaker not in spk2type_str:
                        subtype = 'N/A'
                    else:
                        subtype=spk2type_str[speaker]
                    label_data = [aq,subtype,group,sev_cat]
                else:
                    label_data = ['Control' for i in range(3)]
                    label_data+=[0]

                csv_data = meta_data + label_data
                csv_writer.writerow(csv_data)


            # if utt_id == 'BU09a-116':
            #     exit()

# Scripts
def segment_scripts_data(root_dir,seg_file,target_dir,stype,seg_wav_bool):
    '''
    segments wavs and store new wav_path
    return id2wav which has {utt_id: (wav_path,dur,HC_AB)}
    '''
    seg_wav_dir = f"{target_dir}/wavs"
    if not os.path.exists(seg_wav_dir):
        os.makedirs(seg_wav_dir)

    # owav
    scp=f"{root_dir}/owav.scp"

    # create spkr->src wav dict
    spkr_src_wav_dict={}
    with open(scp, 'r') as r:
        lines = r.readlines()
        # for line in tqdm(lines, total=len(lines)):
        for line in lines:
            spkr_id = line.split()[0]
            wav_path = line.split()[1]
            spkr_src_wav_dict[spkr_id]=wav_path


    missed_files=[]
    id2wav = {}
    with open(seg_file, 'r') as r:
        # get control/aphasia and speaker
        lines = r.readlines()
        for line in lines:
            utt_id = line.split()[0]
            speaker = line.split()[1]
            start_t = float(line.split()[2])
            end_t = float(line.split()[3])
            dur = end_t-start_t


            src_wav_path = spkr_src_wav_dict[speaker]

            new_seg_wav_path = f"{seg_wav_dir}/{utt_id}.wav"


            if seg_wav_bool:
                # sox segmentation
                cmd = ['sox', src_wav_path, new_seg_wav_path, 'trim', f"{start_t}", f"={end_t}"]
                list_files = subprocess.run(cmd)
                # print(list_files)
                # assert list_files.returncode == 0, f"sox error {utt_id}: {start_s} = {end_s}"
                if list_files.returncode != 0:
                    missed_files.append(utt_id)
                    continue
            
            
            id2wav[utt_id] = (new_seg_wav_path,dur)
        # print(cmd)
        # exit()

    # output missed files
    if len(missed_files) > 1:
        with open(f'missed_files_{stype}.txt', 'w') as w:
            for line in missed_files:
                w.write(f"line\n")

    return id2wav

def get_word_labels(duc_src_dir):
    aphasia_wrd_labels = f"{duc_src_dir}/wrd_labels"

    utt2wrd_label = {}
    labels = []
    with open(aphasia_wrd_labels, 'r') as r:
        for line in r.readlines():
            line = line.strip()
            utt_id = line.split()[0]

            word_label_seq=[]
            for c in line.split()[1:]: 
                if c == 'x@n':
                    tok = "n"
                elif c == 'C':
                    tok = c.lower()
                else:
                    tok = c.split(":")[0]
                word_label_seq.append(tok)


            # c, p, or n
            word_label_seq = ['c' if w not in ['p', 'n', 's'] else w for w in word_label_seq]
            
            utt2wrd_label[utt_id] = word_label_seq
            labels+=word_label_seq
    
    count = Counter(labels)
    print(count)
    total = sum(count.values())
    print(f"p: {count['p']/total}")
    print(f"n: {count['n']/total}")
    print(f"s: {count['s']/total}")
    # exit()

    return utt2wrd_label

def add_word_label(src_csv, tgt_csv, utt2wrd_label, para_type,stype):
    df = pd.read_csv(src_csv)
    og_num_samples = df.shape[0]


    paraphasia_list = []
    drop_cols = []
    aug_para_wrds= []
    

    # filter paraphasias
    for i,row in df.iterrows():
        if row['ID'] not in utt2wrd_label or len(row['wrd'].split()) != len(utt2wrd_label[row['ID']]):
            drop_cols.append(i)
        else:
            paraphasia_list.append(" ".join(utt2wrd_label[row['ID']]))

            
            if para_type=='multi':
                aug_para_wrds.append(" ".join([f"{word}/{para}" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])]))
            elif para_type == 'pn':
                aug_para_wrds.append(" ".join([f"{word}/C" if para == 'c' else f"{word}/P" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])]))
            elif para_type == 'p':
                aug_para_wrds.append(" ".join([f"{word}/P" if para == 'p' else f"{word}/C" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])]))
            elif para_type == 'n':
                aug_para_wrds.append(" ".join([f"{word}/P" if para == 'n' else f"{word}/C" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])]))
            elif para_type == 's':
                aug_para_wrds.append(" ".join([f"{word}/S" if para == 's' else f"{word}/C" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])]))

    # drop rows
    df = df.drop(drop_cols)
    df['paraphasia'] = paraphasia_list
    df['aug_para'] = aug_para_wrds


    new_num_samples = df.shape[0]

    # print(f"{stype}: {og_num_samples} -> {new_num_samples}")
    df.to_csv(tgt_csv)


def prepare_Scripts_data():
    root_dir = "/z/mkperez/AphasiaBank/kd_updated_para"
    duc_src_dir=f"{root_dir}/Scripts" # need segments and text file
    # duc_src_dir=f"{root_dir}/Scripts_script_ind"
    target_root="/z/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"
    # target_root="/z/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_noFillers"
    seg_wav_bool = True

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



    utt2wrd_label = get_word_labels(duc_src_dir)

    for fold_dir in tqdm(os.listdir(f"{duc_src_dir}/CV")):
        # make target directory
        target_dir = f"{target_root}/{fold_dir}"
        os.makedirs(target_dir, exist_ok=True)
        # make train, dev, test csvs
        for stype in ['dev','train', 'test']:
            csv_path = f"{target_dir}/{stype}.csv"
            csv_file = open(csv_path, 'w')
            csv_writer = csv.writer(csv_file)
            # headerdata = ['wrd', 'dataset', 'spk_id', 'ID', 'wav', 'duration','severity','aphasia_type','severity_cat','group','speaker']
            headerdata = ['wrd', 'spk_id', 'ID', 'wav', 'duration']
            csv_writer.writerow(headerdata)

            seg_file = f"{duc_src_dir}/CV/{fold_dir}/{stype}/segments"
            text_file = f"{duc_src_dir}/CV/{fold_dir}/{stype}/text"
            id2wav = segment_scripts_data(duc_src_dir,seg_file,target_root,stype,seg_wav_bool)
            id2text = read_text_data(text_file)

            # contains both control and aph speakers
            for utt_id in id2wav.keys():
                group = re.split(r'(\d+)', utt_id)[0]
                speaker = utt_id.split("-")[0]



                # wrd,dataset,spk_id,ID,wav,duration,severity,aphasia_type,severity_cat,group,severity_cat_num
                meta_data = [id2text[utt_id],speaker,utt_id,id2wav[utt_id][0],round(id2wav[utt_id][1],2)]
                csv_data = meta_data
                csv_writer.writerow(csv_data)
            csv_file.close()

            # after writing initial csv add word labels
            for para_type in ['pn','p','n','s','multi']:
                tgt_csv = f"{target_dir}/{stype}_{para_type}.csv"
                add_word_label(csv_path, tgt_csv, utt2wrd_label, para_type, stype)



if __name__ == "__main__":
    # prepare_duc_data()
    prepare_Scripts_data()




'''
Supplement existing data with word_labels
update data w.o neologistic 
'''
import os
import pandas as pd
from collections import Counter

if __name__ == "__main__":
    kaldi_dir = "kd_updated_para"
    aphasia_wrd_labels = f"/y/mkperez/AphasiaBank/{kaldi_dir}/Aphasia/wrd_labels"
    control_wrd_labels = f"/y/mkperez/AphasiaBank/{kaldi_dir}/Control/wrd_labels"
    og_data_dir = "/z/mkperez/speechbrain/AphasiaBank/data/no_paraphasia"
    new_data_dir = f"{og_data_dir}/mtl_paraphasia"

    utt2wrd_label = {}
    labels = []
    for fpath in [aphasia_wrd_labels, control_wrd_labels]:
        with open(fpath, 'r') as r:
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

                # word_label_seq = [c if c == 'C' else c.split(":")[0] for c in line.split()[1:]]
                # word_label_seq 

                # c, p, or n
                word_label_seq = ['c' if w not in ['p', 'n', 's'] else w for w in word_label_seq]
                
                utt2wrd_label[utt_id] = word_label_seq
                labels+=word_label_seq
    print(f"labels: {Counter(labels)}")


    # data dirs for paraphasia
    os.makedirs(new_data_dir,exist_ok=True)

    # make new paraphasia csvs
    for para_type in ['pn','p','n','s','multi']:
        print(f"paraphasia: {para_type}")
        for stype in ['train', 'dev','test']:
            og_file = f"{og_data_dir}/{stype}.csv"
            df = pd.read_csv(og_file)
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
                        aug_para_wrds.append(" ".join([f"{word}/P" if para == 'n' else f"{word}/C" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])]))



    
            # drop rows
            df = df.drop(drop_cols)
            df['paraphasia'] = paraphasia_list
            df['aug_para'] = aug_para_wrds


            new_num_samples = df.shape[0]

            print(f"{stype}: {og_num_samples} -> {new_num_samples}")
            df.to_csv(f"{new_data_dir}/{stype}_{para_type}.csv")


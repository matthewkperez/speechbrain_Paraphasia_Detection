import pandas as pd

WRD_LABELS_FRID = "/z/mkperez/AphasiaBank/kd_updated_para/Scripts/wrd_labels"
KALDI_DIR = "/z/mkperez/AphasiaBank/kd_updated_para/Aphasia+Control/ASR_fold_5"

WRD_LABELS_PROTO_CONTROL = "/z/mkperez/AphasiaBank/kd_updated_para/Control/wrd_labels"
WRD_LABELS_PROTO_APHASIA= "/z/mkperez/AphasiaBank/kd_updated_para/Aphasia/wrd_labels"


def wrd_label_analysis_Frid():
    # output pct of utt and word paraphasia labels
    with open(WRD_LABELS_FRID, 'r') as r:
        lines = r.readlines()
        word = {'n': [], 'pn': [], 'p':[]}
        utt = {'n': [], 'pn': [], 'p':[]}
        for line in lines:
            utt_id = line.split()[0]
            para = line.split()[1:]

            para_n = [1 if p.startswith('n') else 0 for p in para]
            para_p = [1 if p.startswith('p') else 0 for p in para]
            para_pn = [1 if p.startswith('p') or p.startswith('n') else 0 for p in para]


            utt['n'].append(max(para_n))
            utt['pn'].append(max(para_pn))
            utt['p'].append(max(para_p))

            word['n'].extend(para_n)
            word['pn'].extend(para_pn)
            word['p'].extend(para_p)


    for para in ['n', 'pn', 'p']:
        pct = sum(utt[para]) / len(utt[para])
        print(f"Utt-{para}: {pct}")


        pct = sum(word[para]) / len(word[para])
        print(f"Word-{para}: {pct}")

def wrd_label_analysis_Proto():
    # Create wrd_label counts
    control_paraphasia_count=0
    control_count=0
    with open(WRD_LABELS_PROTO_CONTROL, 'r') as r:
        for line in r.readlines():
            for w in line.split()[1:]:
                if w != 'C':
                    control_paraphasia_count+=1
                else:
                    control_count+=1
    # print(control_paraphasia_count, control_count)
    # exit()

    
    # Aphasia
    spk2aq = load_aq()
    utt2spk = load_spks()
    aphasia_count={}
    with open(WRD_LABELS_PROTO_APHASIA, 'r') as r:
        for line in r.readlines():
            uid = line.split()[0]
            
            if uid not in utt2spk:
                continue
            spk = utt2spk[uid]
            if spk not in spk2aq:
                continue

            aq = spk2aq[spk]

            for w in line.split()[1:]:
                if w != 'C':
                    # counter
                    if aq not in aphasia_count:
                        aphasia_count[aq] = 0
                    aphasia_count[aq]+=1
    
    tot = 0
    for k,v in aphasia_count.items():
        print(f"{k}: {v}")
        tot+=v
    print(f"tot: {tot}")



def load_durs():
    utt2dur = {}
    for stype in ['train', 'dev', 'test']:
        durfile = f'{KALDI_DIR}/{stype}/utt2dur'
        with open(durfile, 'r') as r:
            lines = r.readlines()
            for line in lines:
                utt = line.split()[0]
                utt2dur[utt] = float(line.split()[1])
    return utt2dur
    
def load_spks():
    utt2spk = {}
    for stype in ['train', 'dev', 'test']:
        durfile = f'{KALDI_DIR}/{stype}/utt2spk'
        with open(durfile, 'r') as r:
            lines = r.readlines()
            for line in lines:
                utt = line.split()[0]
                utt2spk[utt] = line.split()[1]
    return utt2spk

def load_aq():
    spk2aq = {}
    filepath = "/z/public/data/AphasiaBank/spkr_info/updated_scores.xlsx"
    df = pd.read_excel(filepath,sheet_name="Time 1")
    df_r = pd.read_excel(filepath,sheet_name="Repeats")
    
    # merged_df = pd.merge(df,df_r,on='Participant ID')
    merged_df = pd.concat([df,df_r])
    merged_df = merged_df[['Participant ID', 'WAB AQ']].dropna(axis=0)
    merged_df = merged_df[merged_df['WAB AQ'] != 'U'] 
    merged_df['WAB AQ'] = merged_df['WAB AQ'].astype(float)

    bins = [0, 25, 50, 75, 100]
    labels = ['V.Sev.', 'Sev', 'Mod', 'Mild']
    merged_df['AQ_str'] = pd.cut(merged_df['WAB AQ'], bins=bins, labels=labels, right=False)
    # print(merged_df)
    # exit()
    
    spk2aq = {row['Participant ID']:row['AQ_str'] for i,row in merged_df.iterrows()}

    return spk2aq

def compute_control_dur():
    utt2dur_path = "/z/mkperez/AphasiaBank/kd_updated_para/Control/utt2dur"
    tot_dur = 0
    with open(utt2dur_path,'r') as r:
        for line in r.readlines():
            dur = float(line.split()[1])
            tot_dur+=dur

    return tot_dur

def compute_duration_stats():
    utt2dur = load_durs()
    utt2spk = load_spks()
    spk2aq = load_aq()
    control_dur = compute_control_dur()

    aq2dur={}
    for uid, dur in utt2dur.items():
        spk = utt2spk[uid]
        if spk not in spk2aq:
            continue
        aq = spk2aq[spk]
        if aq not in aq2dur:
            aq2dur[aq]=0
        aq2dur[aq]+=dur

    aq2dur['control'] = control_dur
    sum_dur = 0
    for k,v in aq2dur.items():
        print(f"{k}: {v/3600}")
        sum_dur+=v
    print(f"Total: {sum_dur/3600}")
        

if __name__ == "__main__":
    # wrd_label_analysis()
    # compute_duration_stats()
    wrd_label_analysis_Proto()
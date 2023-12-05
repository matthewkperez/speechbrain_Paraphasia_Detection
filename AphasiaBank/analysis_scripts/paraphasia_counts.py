'''
Get statistical breakdown of paraphasia counts for each speaker Frid
'''
from collections import Counter
import pandas as pd


def get_word_labels(duc_src_dir):
    aphasia_wrd_labels = f"{duc_src_dir}/wrd_labels"

    utt2wrd_label = {}
    labels = []
    df_lst=[]
    with open(aphasia_wrd_labels, 'r') as r:
        for line in r.readlines():
            line = line.strip()
            utt_id = line.split()[0]
            spkr = utt_id.split("_")[0]



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
            
            counts = Counter(word_label_seq)
            
            for para in ['p', 'n', 's']:
                if para not in counts:
                    counts[para] = 0


            df_loc = pd.DataFrame({
                'spkr': [spkr],
                'p': [counts['p']],
                'n': [counts['n']],
                'words': [len(word_label_seq)],
                'utt': [1]
            })
            df_lst.append(df_loc)

    df = pd.concat(df_lst)
    print(df)
    grouped_df = df.groupby('spkr').sum()
    print(grouped_df)

            
            # utt2wrd_label[utt_id] = word_label_seq
    #         labels+=word_label_seq
    
    # count = Counter(labels)
    # print(count)
    # total = sum(count.values())
    # print(f"p: {count['p']/total}")
    # print(f"n: {count['n']/total}")
    # print(f"s: {count['s']/total}")

if __name__ == "__main__":
    ROOT_DIR="/z/mkperez/AphasiaBank/kd_updated_para/Scripts"
    get_word_labels(ROOT_DIR)
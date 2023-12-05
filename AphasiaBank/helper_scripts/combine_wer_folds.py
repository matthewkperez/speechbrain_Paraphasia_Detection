'''
Combine wer files across Folds
Used for SLP analysis
'''


EXP_DIR="/home/mkperez/speechbrain/AphasiaBank/results/Frid-vocab-sweep/2000-unigram-pn"
OUT_FILE = f"{EXP_DIR}/results/wer_comb.txt"

all_lines=[]
for fold in range(1,13):
    loc_wer_file = f"{EXP_DIR}/Fold-{fold}/awer_para.txt"
    with open(loc_wer_file,'r') as r:
        lines = r.readlines()
        all_lines+=lines
        print(f"len: {len(all_lines)}")


with open(OUT_FILE, 'w') as w:
    for line in all_lines:
        w.write(line)
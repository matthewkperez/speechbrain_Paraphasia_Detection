'''
Given wer.txt file
Compute overall wer
Compute breakdown for severities
'''



def individual_Fold_WER(wer_file):
    err = 0
    tot = 0
    with open(wer_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('P') and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                err += int(line.split()[4])
                tot += int(line.split()[6][:-1])

    
    return err, tot


def wer_severity(WER_FILE):
    SEV = "spk2sevbin"
    spk2sev = {}
    with open(SEV, 'r') as r:
        for line in r.readlines():
            spk2sev[line.split()[0]] = line.split()[1]

            
    print(spk2sev)


    err = 0
    tot = 0
    with open(WER_FILE, 'r') as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('P') and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                err += int(line.split()[4])
                tot += int(line.split()[6][:-1])
    

    
    

if __name__ == "__main__":

    WER_FILE="/home/mkperez/speechbrain/AphasiaBank/results/unfrozen_accessible_ASR/hubert-large-ls960-ft/freeze-False/wer.txt"
    wer_severity(WER_FILE)


    # err, tot = individual_Fold_WER(WER_FILE)
    # print(f"WER: {err/tot}")
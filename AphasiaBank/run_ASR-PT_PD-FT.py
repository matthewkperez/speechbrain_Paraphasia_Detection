'''
Run through all folds of Fridriksson
Use PT S2S ASR-only model (Proto)
S2S model jointly optimized for both ASR and paraphasia detection
'''
import os
import shutil
import subprocess
import time
import datetime
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import Counter
import pickle
from scipy import stats
import re
import socket
from tqdm import tqdm

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]



TOT_EPOCHS=100

def train_log_check(train_log_file, last_epoch):
    with open(train_log_file, 'r') as file:
        last_line = file.readlines()[-1].strip()

        if int(last_line.split()[2]) == last_epoch:
            return True
        else:
            print(f"Error, last line epoch = {last_line.split()[2]}")
            return False
      
def compute_maj_class(fold,para_type):
    # compute majority class for naive baseline
    data = f"/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para/Fold_{fold}/train_{para_type}.csv"
    df = pd.read_csv(data)
    PARA_DICT = {'P':1, 'C':0}
    utt_tr = []
    word_tr = []
    for utt in df['aug_para']:
        utt_arr = []
        for p in utt.split():
            utt_arr.append(PARA_DICT[p.split("/")[-1]])
        utt_tr.append(max(utt_arr))
    
    utt_counter = Counter(utt_tr)
    maj_class_utt = utt_counter.most_common()[0][0]
    return maj_class_utt

def compute_recall_n(true_labels, predicted_labels, n=0):
    assert len(true_labels) == len(predicted_labels), "Length of true_labels and predicted_labels must be the same"

    TP = 0
    FN = 0

    for i, (true_label, predicted_label) in tqdm(enumerate(zip(true_labels, predicted_labels))):
        # Create a neighborhood slice
        neighborhood = predicted_labels[max(i-n, 0):min(i+n+1, len(predicted_labels))]


        # print(f"index range: {max(i-n, 0)} - {min(i+n+1, len(predicted_labels))} ")
        # print(f"true: {true_label} | pred: {neighborhood}")
        # print(f"list: {[predicted_labels[max(i-n, 0):min(i+n+1, len(predicted_labels))] == true_label]}")
        if true_label == 1:
            if any(label == true_label for label in neighborhood):
                TP += 1
            else:
                FN += 1

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

def compute_f1_score_recall_n(true_labels, predicted_labels, n=0):
    assert len(true_labels) == len(predicted_labels), "Length of true_labels and predicted_labels must be the same"

    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives

    for i, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
        neighborhood = predicted_labels[max(i-n, 0):min(i+n+1, len(predicted_labels))]

        if true_label == 1:
            if any(label == true_label for label in neighborhood):
                TP += 1
            else:
                FN += 1
        # elif true_label != predicted_label:
        elif not any(label == true_label for label in neighborhood):
            FP += 1

    # Calculating precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculating F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score, recall

def extract_wer(wer_file):
    with open(wer_file, 'r') as r:
        first_line = r.readline()
        wer = float(first_line.split()[1])
        err = int(first_line.split()[3])
        total = int(first_line.split()[5][:-1])
        
        wer_details = {'wer': wer, 'err': err, 'tot': total}


    return wer_details

def extract_utt_F1(wer_file):
    with open(wer_file, 'r') as r:
        lines = r.readlines()

        # Find the index to start extracting lines
        start_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "and  ; hypothesis ; on ; the ; third ; <eps>":
                start_index = i + 1  # Skip the current line
                break
        
        # Extract reference and hypothesis lines
        reference_lines = []
        hypothesis_lines = []
        utt_id2pred = {}

        PARA_KEY={'C':0,'P':1}
        tracker=0
        
        for line in lines[start_index:]:
            line = line.strip()
            # print(f"line: {line}")
            if line.startswith("P") and len(line.split())==14:
                # new sample
                utt_id = line.split()[0][:-1]
                utt_id2pred[utt_id]=None
                tracker=1
            elif tracker==1:
                # print(f"line: {[w.strip() for w in line.split(';')]}")
                para_line = [PARA_KEY[p] for w in line.split(';') if '/' in w for p in w.strip().split("/")[-1]]
                # print(f"para_line: {para_line} | max: {max(para_line)}\n")
                reference_lines.append(max(para_line))
                tracker=2
            elif tracker==2:
                # insertion/del/sub
                tracker=3
            elif tracker==3:
                para_line = [PARA_KEY[p] for w in line.split(';') if '/' in w for p in w.strip().split("/")[-1]]
                hypothesis_lines.append(max(para_line))
                utt_id2pred[utt_id]=max(para_line)
                tracker=0

        # Compute F1
        return reference_lines, hypothesis_lines, utt_id2pred

def extract_pwer(wer_file):
    # extract wer for paraphasic words only
    err = 0
    tot = 0
    with open(wer_file, 'r') as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                # print(line)
                # print(new)
                # exit()
                gt_dict = { i:w.split("/")[0] for i,w in enumerate(words) if w.endswith("/P")}
                switch=2
            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                tot += len(gt_dict)
                for i,w in enumerate(words):
                    if i in gt_dict and gt_dict[i]!=w.split("/")[0]:
                        err+=1
                        
                switch = 0


    return err, tot

def _helper_paraphasia_align(para_words, ref_align):
    # Use ref_align as alignment. para_words provide labels
    # ref_align = [words,] ASR
    # para_words = [word/P, word/C] MTL
    # nbest = neighbors L/R to consider for recall

    ref_index = 0
    para_index = 0
    para_seq = []
    PARA_DICT={'P':1, 'C':0}
    while ref_index < len(ref_align) and para_index < len(para_words):
        para_word = para_words[para_index].split("/")[0]


        if para_word == '<eps>' and ref_align[ref_index] == '<eps>':
            para_seq.append(0)
            ref_index+=1
            para_index+=1

        elif ref_align[ref_index] == '<eps>':
            # add control
            para_seq.append(0)
            ref_index+=1
        
        elif para_word == '<eps>':
            para_index+=1

        elif ref_align[ref_index] == para_word:
            para_seq.append(PARA_DICT[para_words[para_index].split("/")[1]])
            ref_index+=1
            para_index+=1

    # check for extra padding in alignment
    while ref_index < len(ref_align):
        if ref_align[ref_index] == '<eps>':
            # add control
            para_seq.append(0)
            ref_index+=1

    return para_seq

                            

def extract_word_level_paraphasias(wer_file, awer_file):
    # WER alignment
    utt2gt = {}
    utt2pred = {}
    with open(wer_file, 'r') as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                utt2gt[utt_id] = words
                switch=2
            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                utt2pred[utt_id] = words
                switch = 0



    # AWER
    # extract y_true and y_pred for recall and f1-score
    # treat <eps> as control
    y_true = []
    y_pred = []
    with open(awer_file, 'r') as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                ref_align = utt2gt[utt_id]
                # gt = _helper_paraphasia_align(words, ref_align)
                gt = [0 if w=='<eps>' or w.split("/")[1] == 'C' else 1 for w in words]
                y_true.extend(gt)
                switch = 2

                # print(f"gt w: {words}")
                # print(f"ref_align: {ref_align}")
                # print(f"seq: {gt}")

            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                ref_align = utt2pred[utt_id]
                # pred = _helper_paraphasia_align(words, ref_align)
                pred = [0 if w=='<eps>' or w.split("/")[1] == 'C' else 1 for w in words]
                y_pred.extend(pred)
                switch = 0
                # print(f"pred: {line}")
                # print(pred)

                
                # print(f"pred w: {words}")
                # print(f"ref_align: {ref_align}")
                # print(f"seq: {pred}")
                assert len(pred) == len(gt)
                # exit()


    return y_true, y_pred

def get_metrics(fold_dir):
    wer_file = f"{fold_dir}/asr_wer.txt"
    wer_details = extract_wer(wer_file)

    baseline_awer_file = f"{fold_dir}/awer_baseline.txt"
    baseline_awer = extract_wer(baseline_awer_file)

    seq2seq_awer_file = f"{fold_dir}/awer_para.txt"
    seq2seq_awer = extract_wer(seq2seq_awer_file)

    train_log_file = f"{fold_dir}/train_log.txt"
    # assert train_log_check(train_log_file, TOT_EPOCHS), f"Error on {fold_dir}"

    # paraphasia WER
    pwer_err, pwer_tot = extract_pwer(seq2seq_awer_file)

    word_y_true, word_y_pred = extract_word_level_paraphasias(wer_file, seq2seq_awer_file)
    # 

    # compute utt-AWER
    y_true, y_pred, utt_id2f1pred = extract_utt_F1(seq2seq_awer_file)
    
    result_df = pd.DataFrame({
        'wer-err': [wer_details['err']],
        'wer-tot': [wer_details['tot']],
        'baseline_awer-err': [baseline_awer['err']],
        'baseline_awer-tot': [baseline_awer['tot']],
        's2s_awer-err': [seq2seq_awer['err']],
        's2s_awer-tot': [seq2seq_awer['tot']],
        'pwer-err': [pwer_err],
        'pwer-tot': [pwer_tot],
    })
    return result_df, y_true, y_pred, utt_id2f1pred, word_y_true, word_y_pred
    
def clean_FT_model_save(path):
    # keep only 1 checkpoint, remove optimizer
    save_dir = f"{path}/save"
    abs_directory = os.path.abspath(save_dir)


    files = os.listdir(abs_directory)

    # Filter files that start with 'CKPT'
    ckpt_files = [f for f in files if f.startswith('CKPT')]

    # If no CKPT files, return
    if not ckpt_files:
        print("No CKPT files found.")
        return

    # Sort files lexicographically, this works because the timestamp format is year to second
    ckpt_files.sort(reverse=True)

    # The first file in the list is the latest, assuming the naming convention is consistent
    latest_ckpt = ckpt_files[0]
    print(f"Retaining the latest CKPT file: {latest_ckpt}")


    # Remove all other CKPT files
    for ckpt in ckpt_files[1:]:
        shutil.rmtree(os.path.join(abs_directory, ckpt))
        print(f"Deleted CKPT file: {ckpt}")

    # remove optimizer
    optim_file = f"{abs_directory}/{latest_ckpt}/optimizer.ckpt"
    os.remove(optim_file)


    

def change_yaml(yaml_src,yaml_target,data_fold_dir,frid_fold,para_type,output_neurons,output_dir,w2v_model):
    # copy src to tgt
    shutil.copyfile(yaml_src,yaml_target)

    # edit target file
    train_flag = True
    reset_LR = True # if true, start lr with init_LR
    output_dir = f"{output_dir}/Fold-{frid_fold}"
    lr = 5.0e-4

    if 'wavlm' in w2v_model:
        w2v_hub = f"microsoft/{w2v_model}"
    else:
        w2v_hub = f"facebook/{w2v_model}"

    
    
    # copy original file over to new dir
    print(f"output dir: {output_dir}")
    model_dict = {'wavlm-large':'wavlm', 'wav2vec2-large-960h-lv60-self':'w2v','hubert-large-ls960-ft':'hubert'}
    # base_model = f"results/Table-1_verbatim/ES_S2S-{model_dict[w2v_model]}-Transformer-{output_neurons}"

    # T3
    base_model = f"results/Table3/ASR-only-Proto/S2S-{model_dict[w2v_model]}-Transformer-{output_neurons}"
    if not os.path.exists(output_dir):
        print("copying dir")
        print(f"source: {base_model}")
        shutil.copytree(base_model,output_dir, ignore_dangling_symlinks=True)

        clean_FT_model_save(output_dir)

        
        
    # replace with raw text
    with open(yaml_target) as fin:
        filedata = fin.read()
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_fold_dir}")
        filedata = filedata.replace('train_flag_PLACEHOLDER', f"{train_flag}")
        filedata = filedata.replace('FT_start_PLACEHOLDER', f"{reset_LR}")
        filedata = filedata.replace('epochs_PLACEHOLDER', f"{TOT_EPOCHS}")
        filedata = filedata.replace('frid_fold_PLACEHOLDER', f"{frid_fold}")
        filedata = filedata.replace('PARA_PLACEHOLDER', f"{para_type}")
        filedata = filedata.replace('output_PLACEHOLDER', f"{output_dir}")
        filedata = filedata.replace('output_neurons_PLACEHOLDER', f"{output_neurons}")
        filedata = filedata.replace('lr_PLACEHOLDER', f"{lr}")

        filedata = filedata.replace('w2v_model_PLACEHOLDER', f"{w2v_model}")
        filedata = filedata.replace('w2v_hub_PLACEHOLDER', f"{w2v_hub}")
        filedata = filedata.replace('MTL_bool_PLACEHOLDER', "False")


        with open(yaml_target,'w') as fout:
            fout.write(filedata)

    return output_dir

if __name__ == "__main__":
    DATA_ROOT = "/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

    TRAIN_FLAG = False
    EVAL_FLAG = True
    PARA_TYPE=['pn','p','n'][0]
    OUTPUT_NEURONS=500
    W2V_MODEL=['wavlm-large', 'wav2vec2-large-960h-lv60-self','hubert-large-ls960-ft'][0]
    EXP_DIR = f"results/Table2/ASR-PT_PD-FT/{W2V_MODEL.split('-')[0]}/{PARA_TYPE}-{OUTPUT_NEURONS}-unigram"

    # for T3
    # EXP_DIR = f"results/Table3/FT-MTL-Script/{W2V_MODEL.split('-')[0]}/{PARA_TYPE}-{OUTPUT_NEURONS}-unigram"


    if TRAIN_FLAG:
        yaml_src = "/home/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/AWER/base_SSL_transformer_best_word.yml"
        yaml_target = "/home/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/AWER/SSL_transformer_best_word.yml"
        start = time.time()
        
        i=1
        count=0
        while i <=12:
            data_fold_dir = f"{DATA_ROOT}/Fold_{i}"

            change_yaml(yaml_src,yaml_target,data_fold_dir,i,PARA_TYPE,OUTPUT_NEURONS,EXP_DIR,W2V_MODEL)

            # # launch experiment
            # multi-gpu
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            # cmd = ['python', '-m', 'torch.distributed.launch', '--nproc_per_node=1',
            #        '--master_port', str(port), 
            #     'Fridriksson_AWER.py', f'{yaml_target}',
            #     '--distributed_launch', '--distributed_backend=nccl', '--find_unused_parameters']
            port = find_free_port()  # Get a free port.
            print(f"free port: {port}")
            cmd = ['torchrun', '--nproc_per_node=1',
                   f'--master_port={str(port)}', 
                'Fridriksson_AWER.py', f'{yaml_target}',
                '--distributed_launch', '--distributed_backend=nccl', '--find_unused_parameters']
            
            # # single gpu
            # env = os.environ.copy()
            # env['CUDA_VISIBLE_DEVICES'] = '0'
            # cmd = ['python', 'Fridriksson_AWER.py', f'{yaml_target}']
            p = subprocess.run(cmd, env=env)

            # p = subprocess.run(cmd)
            count+=1
            print(f"p.returncode: {p.returncode} | retry: {count}")
            # exit()
            

            if p.returncode == 0:
                i+=1


        end = time.time()
        elapsed = end-start
        print(f"Total Train runtime: {datetime.timedelta(seconds=elapsed)}")

    ##  Stat computation
    if EVAL_FLAG:
        results_dir = f"{EXP_DIR}/results"
        os.makedirs(results_dir, exist_ok=True)

        df_list = []
        y_true = []
        y_pred = []
        word_y_true = []
        word_y_pred = []
        y_maj_class = []
        utt_id2f1pred = {}
        for i in range(1,13):
            Fold_dir = f"{EXP_DIR}/Fold-{i}"
            result_df, y_true_loc, y_pred_loc, utt_id2f1pred_loc, word_y_true_loc, word_y_pred_loc = get_metrics(Fold_dir)
            # print(f"Frid-{i}\ntrue:{y_true_loc}\npred:{y_pred_loc}")
            # exit()

            # Naive approach
            maj_class_utt = compute_maj_class(i,PARA_TYPE)
            y_maj_class.extend([maj_class_utt for i in y_pred_loc])
                
            # Utt-F1
            y_true.extend(y_true_loc)
            y_pred.extend(y_pred_loc)
            df_list.append(result_df)
            utt_id2f1pred.update(utt_id2f1pred_loc)
            word_y_true.extend(word_y_true_loc)
            word_y_pred.extend(word_y_pred_loc)
            # print(f"{i}: {len(utt_id2f1pred_loc.keys())}")
            # print(f"{i}: {utt_id2f1pred_loc.keys()}")
            # exit()
        df = pd.concat(df_list)


        base_utt_f1 = f1_score(y_true, y_maj_class, average='macro')
        utt_f1 = f1_score(y_true, y_pred, average='macro')
        utt_precision = precision_score(y_true, y_pred, average='macro')
        utt_recall = recall_score(y_true, y_pred, average='macro')
        utt_recall_binary = recall_score(y_true, y_pred, average='binary')
        # word_level
        word_level_f1 = f1_score(word_y_true, word_y_pred, average='binary')
        word_level_precision = precision_score(word_y_true, word_y_pred, average='binary')

        # Recall-f1 localization
        zero_f1, zero_recall = compute_f1_score_recall_n(word_y_true, word_y_pred, n=0)
        one_f1, one_recall = compute_f1_score_recall_n(word_y_true, word_y_pred, n=1)
        two_f1, two_recall = compute_f1_score_recall_n(word_y_true, word_y_pred, n=2)

        with open(f"{results_dir}/Frid_metrics.txt", 'w') as w:
            for k in ['wer', 'baseline_awer', 's2s_awer', 'pwer']:
                wer = df[f'{k}-err'].sum()/df[f'{k}-tot'].sum()
                print(f"{k}: {wer}")
                w.write(f"{k}: {wer}\n")
            
            print(f"Utt-F1: {utt_f1}")
            w.write(f"Utt-F1: {utt_f1}\n")
            print(f"Utt-prec: {utt_precision}")
            w.write(f"Utt-prec: {utt_precision}\n")
            print(f"Utt-recall: {utt_recall}")
            w.write(f"Utt-recall: {utt_recall}\n")


            print(f"Utt-recall-binary: {utt_recall_binary}\n")
            w.write(f"Utt-recall-binary: {utt_recall_binary}\n\n")
            print(f"word-f1-binary: {word_level_f1}")
            w.write(f"word-f1-binary: {word_level_f1}\n")
            print(f"word-precision-binary: {word_level_precision}")
            w.write(f"word-prec-binary: {word_level_precision}\n")


            # Recall -word
            print(f"0-word-recall: {zero_recall}")
            w.write(f"0-word-recall: {zero_recall}\n")
            print(f"1-word-recall: {one_recall}")
            w.write(f"1-word-recall: {one_recall}\n")
            print(f"2-word-recall: {two_recall}\n")
            w.write(f"2-word-recall: {two_recall}\n\n")
            
            # F1
            print(f"0-word-f1: {zero_f1}")
            w.write(f"0-word-f1: {zero_f1}\n")
            print(f"1-word-f1: {one_f1}")
            w.write(f"1-word-f1: {one_f1}\n")
            print(f"2-word-f1: {two_f1}\n")
            w.write(f"2-word-f1: {two_f1}\n\n")

        with open(f"{results_dir}/utt2pred_uttf1.pkl", 'wb') as w:
            pickle.dump(utt_id2f1pred,w)

        

        

        



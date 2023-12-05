'''
Analyze Paraphasia performance and WER performance as functions of vocab size
'''
import os
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

RESULTS_DIR="/z/mkperez/speechbrain/AphasiaBank/results/Frid"
ANALYSIS_DIR="/z/mkperez/speechbrain/AphasiaBank/analysis/Frid"
DUC_DIR="/z/mkperez/speechbrain/AphasiaBank/results/Frid/Duc-results"
os.makedirs(ANALYSIS_DIR, exist_ok=True)




def read_metrics_file(metric_file):
    data = {}
    with open(metric_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            key = line.split(": ")[0]
            val = float(line.split(": ")[-1])
            data[key] = val

    return pd.DataFrame(data, index=[0])

def plot_metrics_wer(df,para_type):
    df_melt = pd.melt(df, id_vars=['vocab_size'], value_vars=['s2s_awer', 'baseline_awer'])
    plt.clf()
    ax = sns.lineplot(data=df_melt, x='vocab_size',y='value', hue='variable')
    ax.set(title=f"AWER on Frid Dataset for paraphasias-{para_type}")
    fig = ax.get_figure()
    fig.savefig(f"{ANALYSIS_DIR}/awer.png")

    # df_melt = pd.melt(df, id_vars=['vocab_size'], value_vars=['Utt-F1'])
    df = df.reset_index(drop=True)
    plt.clf()
    ax = sns.lineplot(data=df, x='vocab_size', y='Utt-F1')
    ax.set(title=f"Utt-F1 on Frid Dataset for paraphasias-{para_type}")
    fig = ax.get_figure()
    fig.savefig(f"{ANALYSIS_DIR}/F1.png")
    
def utt_f1_sig_check(paraphasia_type,seq2seq_dir):
    # Load seq2seq output

    df_metrics = read_metrics_file(f'{seq2seq_dir}/Frid_metrics.txt')
    print(f"Metrics:\n{df_metrics}")

    # load seq2seq pkl
    with open(f'{seq2seq_dir}/utt2pred_uttf1.pkl', 'rb') as file:
        s2s_utt2pred = pickle.load(file)
        print(f"s2s_pred: {len(s2s_utt2pred.keys())}")


    with open(f'{DUC_DIR}/f1_outs_{paraphasia_type}.pkl', 'rb') as r:
        duc_results = pickle.load(r)

        duc_pred = duc_results['pred']
        duc_true = duc_results['true']
        duc_ids = duc_results['ids']
        print(f"duc_ids: {len(duc_ids)}")
        s2s_pred = [s2s_utt2pred[i] for i in duc_results['ids']]

        print(f"duc_pred: {duc_pred}")
        print(f"s2s_pred: {s2s_pred}")
        exit()
        t_statistic, p_value = stats.ttest_rel(duc_pred, y_pred)

def main():
    paraphasia_type = "n"
    # WER
    df_list = []
    for dir in os.listdir(RESULTS_DIR):
        if dir[0] in string.digits:
            vocab_size = int(dir.split("-")[0])
            ptype = dir.split("-")[-1]
            if ptype == paraphasia_type:

                results_file = f"{RESULTS_DIR}/{dir}/results/Frid_metrics.txt"
                # print(results_file)
                df_loc = read_metrics_file(results_file)
                df_loc['vocab_size'] = vocab_size
                df_list.append(df_loc)
    
    df = pd.concat(df_list)
    print(df)
    plot_metrics_wer(df,paraphasia_type)

    # S2S_DIR = f"{RESULTS_DIR}/500-unigram-{paraphasia_type}/results"
    # utt_f1_sig_check(paraphasia_type,S2S_DIR)


                

            
    

if __name__ == "__main__":
    main()
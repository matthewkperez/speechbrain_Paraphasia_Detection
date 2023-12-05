'''
Plot recall and f1-score
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

MTL_TXT = "data/MTL-wavlm-transformer-pn-500.txt"
STL_TXT = "data/STL-wavlm-transformer-pn-500.txt"


df_list = []
def barplot(wfile):
    for tfile, t_title in zip([MTL_TXT,STL_TXT], ['MTL', 'STL']):
        with open(tfile, 'r') as r:
            lines = r.readlines()
            file_dict = {}
            for line in lines:
                if '-word-' in line or 'Utt-F1' in line or 'Utt-recall' in line:
                    k = line.split(": ")[0]
                    val = float(line.split(": ")[1])
                    file_dict[k] = val
            
            
            window_size = [0,1,2]
            for w in window_size:
                window = w*2 + 1

                df_loc = pd.DataFrame({
                    'window': [window],
                    'recall': [file_dict[f'{w}-word-recall']],
                    'f1-score': [file_dict[f'{w}-word-f1']],
                    'model': [t_title],
                })
                df_list.append(df_loc)

            # utt-level
            df_loc = pd.DataFrame({
                'window': ['utt'],
                'recall': [file_dict[f'Utt-recall-binary']],
                'f1-score': [file_dict[f'Utt-F1']],
                'model': [t_title],
            })
            df_list.append(df_loc)

    df = pd.concat(df_list)

    df = df.reset_index()

    utt_MTL = df.iloc[3]
    utt_STL = df.iloc[7]

    df = df.drop([3,7])
    print(df)


    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(data=df, x='window',y='recall', hue='model', ax=axes[1])
    axes[1].set_title('Windowed Recall')
    axes[1].tick_params(axis='both', labelsize=13)  # Increase tick label size
    axes[1].set_xlabel('Window Size', fontsize=14)  # Increase x-axis label size
    axes[1].set_ylabel('Recall', fontsize=14)    # Increase y-axis label size
    axes[1].axhline(y=utt_MTL['recall'], color='blue', linestyle='--', linewidth=2)  # Horizontal line
    axes[1].axhline(y=utt_STL['recall'], color='orange', linestyle='--', linewidth=2)  # Horizontal line


    sns.barplot(data=df, x='window',y='f1-score', hue='model', ax=axes[0])
    axes[0].set_title('Windowed F1-score')
    axes[0].tick_params(axis='both', labelsize=13)  # Increase tick label size
    axes[0].set_xlabel('Window Size', fontsize=14)  # Increase x-axis label size
    axes[0].set_ylabel('F1 Score', fontsize=14)    # Increase y-axis label size
    axes[0].axhline(y=utt_MTL['f1-score'], color='blue', linestyle='--', linewidth=2)  # Horizontal line
    axes[0].axhline(y=utt_STL['f1-score'], color='orange', linestyle='--', linewidth=2)  # Horizontal line


    dotted_line_mtl = mlines.Line2D([], [], color='blue', linestyle='--', linewidth=2, label='MTL-max')
    dotted_line_stl = mlines.Line2D([], [], color='orange', linestyle='--', linewidth=2, label='STL-max')
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(dotted_line_mtl)  # Add the dotted line to the existing handles
        handles.append(dotted_line_stl)  # Add the dotted line to the existing handles
        ax.legend(handles=handles, loc='upper left')


    # fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(wfile)
                    





if __name__ == "__main__":
    barplot("barplot_STL_MTL.png")
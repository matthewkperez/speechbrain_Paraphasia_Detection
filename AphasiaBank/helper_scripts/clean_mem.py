import os
import shutil

EXP_DIR="/home/mkperez/speechbrain/AphasiaBank/results/Frid-vocab-sweep"

for sdir in os.listdir(EXP_DIR):
    for i in range(1,13):
        save_dir = f"{EXP_DIR}/{sdir}/Fold-{i}/save"

        shutil.rmtree(save_dir)

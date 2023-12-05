#!/bin/sh

#SBATCH -p spgpu
#SBATCH --mail-type=END
#SBATCH --account=emilykmp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=T3-wavlm-100
#SBATCH --output=batch_logs/Table3/wavlm-100.log
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=45G


# Run with `sbatch ./sbatch.sh`

# We activate our env
cd /home/mkperez/speechbrain/AphasiaBank
source /home/mkperez/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
# set -x
conda activate sb
# module load python/3.10.4
# module load python3.9-anaconda/2021.11


## Proto-Accessible (Table 1)
# MAX MASTER_PORT=65535
export MASTER_PORT=23434
# torchrun --nproc_per_node=1 train.py hparams/Duc_process/base.yaml
# torchrun --nproc_per_node=1 train_whisper.py hparams/Duc_process/whisper.yaml 

## S2S
# torchrun --nproc_per_node=1 train_SSL_transformer.py hparams/Duc_process/SSL_transformer.yml
# torchrun --nproc_per_node=1 train_Transformer.py hparams/Duc_process/transformer.yml

# Table3 - ASR-only
torchrun --nproc_per_node=1 train_SSL_transformer.py hparams/Duc_process/Table1/wavlm.yml
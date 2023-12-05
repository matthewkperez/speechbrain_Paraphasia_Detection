#!/bin/sh

#SBATCH -p spgpu
#SBATCH --mail-type=END
#SBATCH --account=emilykmp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=T3-2000
#SBATCH --output=batch_logs/Table3/S2S/2000.log
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=45G


# Run with `sbatch ./sbatch.sh`

# We activate our env
cd /home/mkperez/speechbrain/AphasiaBank
source /home/mkperez/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
# set -x
conda activate sb

## MTL
## PT w/ proto
# torchrun --nproc_per_node=1 MTL_S2S_Proto.py hparams/Duc_process/MTL_S2S_Proto.yml


## FT w/ scripts
srun python run_Frid_best_word.py

# ASR-only -> FT-PD
# srun python run_ASR-PT_PD-FT.py
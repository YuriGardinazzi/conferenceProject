#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=14:00:00
#SBATCH --mem=400g
#SBATCH --output=rep_output_%j.out
cd /orfeo/LTS/LADE/LT_storage/ygardinazzi/tda_transformers/rep_llama
echo $(date)
source /u/area/ygardinazzi/scratch/miniconda/bin/activate
export TOKENIZERS_PARALLELISM=True
conda activate llama

python extract.py

echo $(date)

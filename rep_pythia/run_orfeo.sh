#!/bin/bash
#SBATCH --job-name=pythia
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=300g
#SBATCH --output=rep_pythia_%j.out
cd /orfeo/LTS/LADE/LT_storage/ygardinazzi/tda_transformers/rep_pythia
echo $(date)
source /u/area/ygardinazzi/scratch/miniconda/bin/activate
#export TOKENIZERS_PARALLELISM=True
conda activate llama

python extract.py

echo $(date)

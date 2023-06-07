#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
# Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install -r $HOME/kbc/requirements.txt
# Copy input files to scratch
cp $HOME/kbc22/data/dev.jsonl "$TMPDIR"
cp $HOME/kbc22/data/train.jsonl "$TMPDIR"
cp $HOME/kbc22/data/dev.pred.jsonl "$TMPDIR"
# output directory on scratch
mkdir -p "$TMPDIR"/data
# Execute baseline.py with the specified arguments
python $HOME/kbc22/train_script.py -m declare-lab/flan-alpaca-large -i "$TMPDIR"/dev.pred.jsonl -o "$TMPDIR"/data/alpaca-v0.jsonl -g 0
# from scratch to home
cp -r "$TMPDIR"/data/alpaca-v0.jsonl $HOME/kbc22/data/

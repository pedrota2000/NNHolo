#!/bin/bash
#SBATCH --job-name=NNHolo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=pytorch_training_%j.out
#SBATCH --error=pytorch_training_%j.err

source /home/pedro/env/bin/activate
module load cuda python3
# Run the first PyTorch training script
python3 train_model1.py --gpu 0 &

# Run the second PyTorch training script
python3 train_model2.py --gpu 0 &
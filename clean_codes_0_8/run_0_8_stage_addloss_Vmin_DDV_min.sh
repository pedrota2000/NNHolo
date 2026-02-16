#!/bin/bash
#SBATCH --job-name=0_8_T2
#SBATCH --ntasks-per-node=1  # Utilize all CPU cores
#SBATCH --partition gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=5-00:00:0
#SBATCH -o /home/ptejerina/NNholo/results/job%j.o
#SBATCH -e /home/ptejerina/NNholo/results/job%j.e
###        #SBATCH --partition=unlimited
#SBATCH --mail-type=ALL       # Send email on job start, end, and failure
#SBATCH --mail-user=pablo.tejerina@icc.ub.edu  # Replace with your email address

module purge
#module load cuda
source /home/ptejerina/holo_env/bin/activate

# Run the first PyTorch training script

# python3 test_0_8_staging_V_DV_DDV_at_min.py
python3 continue_test_0_8_staging_V_DV_DDV_at_min_test_2.py

wait
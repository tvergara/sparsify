#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=48G                                        # Ask for 10 GB of RAM
#SBATCH --time=120:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

# 1. Load the required modules
module --quiet load miniconda/3

# 2. Load your environment
conda activate saes

python3 -m sparsify --data_preprocessing_num_proc 4 --target_model HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints --finetune $SCRATCH/sae-checkpoints/seed1 --freeze_encoder --layer_stride 3 --target_revision step-240000 --save_dir $SCRATCH/saes/finetuning-step-240000

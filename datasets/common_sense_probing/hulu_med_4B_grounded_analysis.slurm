#!/bin/bash -l

#SBATCH --job-name=hulu-med-4B-grounded
#SBATCH --comment="Hulu-Med-4B grounded questions analysis on Cholec80"

#SBATCH --account=vlm-idk
#SBATCH --partition=sporc-gpu

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1

# Activate conda environment
source /home/as5606/miniconda3/etc/profile.d/conda.sh
conda activate hulu-med-clean
export HF_HOME=/shared/rc/vlm-idk/huggingface_cache

# Run the analysis script
cd /home/as5606/Datasets/Cholec_text_grounded_questions
python3 hulu_med_4B_grounded_analysis.py

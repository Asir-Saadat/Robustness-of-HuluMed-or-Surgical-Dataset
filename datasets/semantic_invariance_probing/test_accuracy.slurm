#!/bin/bash -l

#SBATCH --job-name=hulu-med-semantic-accuracy
#SBATCH --comment="Semantic variations accuracy evaluation"

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

# Get model size from command line argument (default to 4B if not specified)
MODEL_SIZE=${1:-4B}

# Run the accuracy evaluation
cd /home/as5606/Datasets/Cholec_text_semantic_similar_questions
python3 test_accuracy.py $MODEL_SIZE

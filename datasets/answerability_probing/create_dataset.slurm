#!/bin/bash -l

#SBATCH --job-name=cholec80-unanswerable
#SBATCH --comment="Generate unanswerable questions for Cholec80"

#SBATCH --account=vlm-idk
#SBATCH --partition=sporc-gpu

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32g

# Activate conda environment (change to your environment if different)
source /home/as5606/miniconda3/etc/profile.d/conda.sh
conda activate llava-train

# Set OpenAI API key (replace with your actual key)
# IMPORTANT: It's better to store this in a .env file or pass via environment
export OPENAI_API_KEY="your-openai-api-key-here"

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set"
    exit 1
fi

# Run the script
cd /home/as5606/Datasets/Cholec_text_hallucinatory_questions
python3 create_dataset.py

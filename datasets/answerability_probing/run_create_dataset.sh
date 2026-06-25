#!/bin/bash

cd /home/as5606/Datasets/Cholec_text_hallucinatory_questions

# Set your OpenAI API key here (or pass via environment)
export OPENAI_API_KEY="os.environ.get("OPENAI_API_KEY")"

# Run the Python script
python3 create_dataset.py

#!/bin/bash

cd /home/as5606/Datasets/Cholec_text_semantic_similar_questions

# Set your OpenAI API key here (or pass via environment)
export OPENAI_API_KEY="your-openai-api-key-here"

# Run the Python script
python3 create_dataset.py

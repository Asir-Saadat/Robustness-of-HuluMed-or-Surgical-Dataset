#!/usr/bin/env python3
"""
Generate VQA answers using OpenAI ChatGPT on Cholec80 test dataset.
"""

import json
import os
import base64
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client

client = OpenAI(api_key="os.environ.get("OPENAI_API_KEY")")


# System prompt
SYSTEM_PROMPT = """\
You are a surgical VQA assistant. Answer questions about laparoscopic cholecystectomy frames.
Answer simply and directly based on what is visible in the image.
"""

# Load test dataset
print("Loading test dataset...")
with open('/home/as5606/Datasets/cholec_formatted_data/cholec80_llava_test.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples\n")

# Store results
results = []

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

print("Generating ChatGPT answers...")
for item in tqdm(test_data, desc="Processing"):
    try:
        item_id = item['id']
        image_path = item['image']
        question = item['conversations'][0]['value'].replace('\n<image>', '').strip()+ " If the question cannot be answered based on the image, say 'I don't know'. Answer simply."
        original_answer = item['conversations'][1]['value']
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Create user message with image
        user_msg = f"Question: {question} Answer simply"
        
        # Generate answer using ChatGPT
        response = client.responses.create(
            model="gpt-5.4-nano",
            instructions=SYSTEM_PROMPT,
            input=user_msg,
        )
        
        chatgpt_answer = response.output_text.strip()
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "question": question,
            "original_answer": original_answer,
            "chatgpt_answer": chatgpt_answer
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/chatGPT/chatgpt_answers_prompt.json'
os.makedirs('/home/as5606/projects/Hulu-Med/chatGPT', exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")

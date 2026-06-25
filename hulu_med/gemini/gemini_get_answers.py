#!/usr/bin/env python3
"""
Generate VQA answers using Google Gemini 3.1 Flash Lite on Cholec80 test dataset.
"""

import json
import sys
from tqdm import tqdm
from google import genai
from google.genai import types

# Initialize client
client = genai.Client(

    api_key="os.environ.get("GOOGLE_API_KEY")"

)

# Load test dataset
print("Loading test dataset...")
with open('/home/as5606/Datasets/cholec_formatted_data/cholec80_llava_test.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples\n")

# Store results
results = []

# MIME type mapping
mime_types = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'webp': 'image/webp'
}

print("Generating Gemini answers...")
for item in tqdm(test_data, desc="Processing"):
    try:
        item_id = item['id']
        image_path = item['image']
        question = item['conversations'][0]['value'].replace('\n<image>', '').strip() + " Answer simply"
        original_answer = item['conversations'][1]['value']
        
        # Read image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get mime type
        image_ext = image_path.lower().split('.')[-1]
        mime_type = mime_types.get(image_ext, 'image/jpeg')
        
        # Generate answer using Gemini
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                question
            ]
        )
        
        gemini_answer = response.text
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "question": question,
            "original_answer": original_answer,
            "gemini_answer": gemini_answer
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/gemini/gemini_test_answers.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")

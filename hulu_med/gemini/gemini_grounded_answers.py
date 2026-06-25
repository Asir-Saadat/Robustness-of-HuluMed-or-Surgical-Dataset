#!/usr/bin/env python3
"""
Generate answers for grounded questions using Google Gemini 3.1 Flash Lite.
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

# Load grounded questions dataset
print("Loading grounded questions dataset...")
with open('/home/as5606/Datasets/Cholec_text_grounded_questions/cholec80_with_grounded_questions.json', 'r') as f:
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

print("Generating Gemini answers for grounded questions...")
for item in tqdm(test_data, desc="Processing"):
    try:
        item_id = item['id']
        image_path = item['image']
        
        # Extract from conversations array
        conversations = item.get('conversations', [])
        original_question = conversations[0]['value'].replace('\n<image>', '').strip() if len(conversations) > 0 else 'Unknown'
        original_answer = conversations[1]['value'] if len(conversations) > 1 else 'Unknown'
        
        # Read image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get mime type
        image_ext = image_path.lower().split('.')[-1]
        mime_type = mime_types.get(image_ext, 'image/jpeg')
        
        # Generate answers for grounded questions
        grounded_qa_with_gemini = []
        for grounded_qa in item.get('grounded_qa_pairs', []):
            question = grounded_qa['question'] + " Answer simply"
            expected_answer = grounded_qa['answer']
            
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
            
            grounded_qa_with_gemini.append({
                "question": grounded_qa['question'],
                "original_answer": expected_answer,
                "gemini_answer": gemini_answer
            })
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "grounded_qa_with_gemini_answers": grounded_qa_with_gemini
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/gemini/gemini_grounded_answers.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")

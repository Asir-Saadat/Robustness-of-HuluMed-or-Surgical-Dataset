#!/usr/bin/env python3
"""
Generate answers for semantic variations using Google Gemini 3.1 Flash Lite.
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


# Load semantic variations dataset
print("Loading semantic variations dataset...")
with open('/home/as5606/Datasets/Cholec_text_semantic_similar_questions/cholec80_with_semantic_variations.json', 'r') as f:
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

print("Generating Gemini answers for semantic variations...")
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
        
        # Generate answers for semantic variations
        semantic_variations_with_gemini = []
        for variation_question in item.get('semantic_variations', []):
            
            # Generate answer using Gemini
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    variation_question + " Answer simply"
                ]
            )
            
            gemini_answer = response.text
            
            semantic_variations_with_gemini.append({
                "question": variation_question,
                "gemini_answer": gemini_answer
            })
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "semantic_variations_with_gemini_answers": semantic_variations_with_gemini
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/gemini/gemini_semantic_variations_answers.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")

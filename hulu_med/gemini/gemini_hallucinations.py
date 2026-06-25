#!/usr/bin/env python3
"""
Generate answers for hallucinatory questions using Google Gemini 3.1 Flash Lite.
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


# Load hallucinatory questions dataset
print("Loading hallucinatory questions dataset...")
with open('/home/as5606/Datasets/Cholec_text_hallucinatory_questions/cholec80_with_new_questions.json', 'r') as f:
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

print("Generating Gemini answers for hallucinatory questions...")
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
        
        # Generate answers for new questions
        new_questions_with_gemini = []
        for new_q in item.get('new_questions', []):
            question_type = new_q['question_type'] if 'question_type' in new_q else new_q.get('type', 'unknown')
            question_text = new_q['question']
            
            # Add instruction prompt
            question_with_prompt = question_text + " If the question cannot be answered based on the image, say 'I don't know'. Answer simply."
            
            # Generate answer using Gemini
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    question_with_prompt
                ]
            )
            
            gemini_answer = response.text
            
            new_questions_with_gemini.append({
                "type": question_type,
                "question": question_with_prompt,
                "gemini_answer": gemini_answer
            })
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "new_questions_with_gemini_answers": new_questions_with_gemini
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/gemini/gemini_hallucinatory_answers.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")

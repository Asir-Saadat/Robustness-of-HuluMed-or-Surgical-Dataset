#!/usr/bin/env python3
"""
Generate answers for grounded questions using OpenAI ChatGPT.
"""

import json
import os
import base64
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="os.environ.get("OPENAI_API_KEY")")


# System prompt
SYSTEM_PROMPT = """\
You are shown ONE laparoscopic cholecystectomy frame.
Answer SIMPLE, visually-grounded questions about surgical tools in the image.
Answer directly based on what you see in the image.
"""

# Load grounded questions dataset
print("Loading grounded questions dataset...")
with open('/home/as5606/Datasets/Cholec_text_grounded_questions/cholec80_with_grounded_questions.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples\n")

# Store results
results = []

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

print("Generating ChatGPT answers for grounded questions...")
for item in tqdm(test_data, desc="Processing"):
    try:
        item_id = item['id']
        image_path = item['image']
        
        # Extract from conversations array
        conversations = item.get('conversations', [])
        original_question = conversations[0]['value'].replace('\n<image>', '').strip() if len(conversations) > 0 else 'Unknown'
        original_answer = conversations[1]['value'] if len(conversations) > 1 else 'Unknown'
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Generate answers for grounded questions
        grounded_qa_with_chatgpt = []
        for grounded_qa in item.get('grounded_qa_pairs', []):
            question = grounded_qa['question'] + " Answer simply"
            expected_answer = grounded_qa['answer']
            
            # Generate answer using ChatGPT
            user_msg = f"Question: {question}"
            
            response = client.responses.create(
                model="gpt-5.4-nano",
                instructions=SYSTEM_PROMPT,
                input=user_msg,
            )
            
            chatgpt_answer = response.output_text.strip()
            
            grounded_qa_with_chatgpt.append({
                "question": grounded_qa['question'],
                "original_answer": expected_answer,
                "chatgpt_answer": chatgpt_answer
            })
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "grounded_qa_with_chatgpt_answers": grounded_qa_with_chatgpt
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/chatGPT/chatgpt_grounded_answers.json'
os.makedirs('/home/as5606/projects/Hulu-Med/chatGPT', exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")

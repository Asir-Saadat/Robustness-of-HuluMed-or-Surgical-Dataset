#!/usr/bin/env python3
"""
Generate answers for hallucinatory questions using OpenAI ChatGPT.
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
Answer questions about the image. If you cannot answer a question based on what
you see in the image, say 'I don't know'.
Answer simply and directly.
"""

# Load hallucinatory questions dataset
print("Loading hallucinatory questions dataset...")
with open('/home/as5606/Datasets/Cholec_text_hallucinatory_questions/cholec80_with_new_questions.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples\n")

# Store results
results = []

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

print("Generating ChatGPT answers for hallucinatory questions...")
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
        
        # Generate answers for new questions
        new_questions_with_chatgpt = []
        for new_q in item.get('new_questions', []):
            question_type = new_q['question_type'] if 'question_type' in new_q else new_q.get('type', 'unknown')
            question_text = new_q['question']
            
            # Add instruction prompt
            question_with_prompt = question_text + " If the question cannot be answered based on the image, say 'I don't know'. Answer simply."
            
            # Generate answer using ChatGPT
            user_msg = f"Question: {question_with_prompt}"
            
            response = client.responses.create(
                model="gpt-5.4-nano",
                instructions=SYSTEM_PROMPT,
                input=user_msg,
            )
            
            chatgpt_answer = response.output_text.strip()
            
            new_questions_with_chatgpt.append({
                "type": question_type,
                "question": question_with_prompt,
                "chatgpt_answer": chatgpt_answer
            })
        
        # Store result
        result = {
            "id": item_id,
            "image": image_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "new_questions_with_chatgpt_answers": new_questions_with_chatgpt
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/projects/Hulu-Med/chatGPT/chatgpt_hallucinatory_answers.json'
os.makedirs('/home/as5606/projects/Hulu-Med/chatGPT', exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")
